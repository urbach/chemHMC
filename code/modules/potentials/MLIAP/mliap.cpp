#include "mliap.hpp"
#include <torch/script.h>
#include "global.hpp"

const std::unordered_map<std::string, int64_t> element_to_atomic = {
    {"H", 1}, {"C", 6},  {"N", 7},  {"O", 8}
};

torch::Tensor labels_to_species(const std::vector<std::string>& labels, torch::Device device) {
    std::vector<int64_t> atomic_numbers;
    atomic_numbers.reserve(labels.size());

    for (const auto& symbol : labels) {
        auto it = element_to_atomic.find(symbol);
        if (it != element_to_atomic.end()) {
            atomic_numbers.push_back(it->second);
        } else {
            throw std::runtime_error("Unknown element symbol: " + symbol);
        }
    }

    torch::Tensor species = torch::from_blob(
        atomic_numbers.data(), {static_cast<int64_t>(atomic_numbers.size())},
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)
    ).clone();

    return species.to(device);
}

void MLIAP::init(const particles_instance& particles) {
    N_at = particles.N;

    // Load module
    device_ = torch::Device(device_str);
    module_ = torch::jit::load(model_path);
    module_.to(device_);
    module_.eval();
    module_.setattr("compute_forces", true);
    module_.setattr("compute_stress", false);
    //module_ = torch::jit::optimize_for_inference(module_);

    // Enable forces
    if (module_.hasattr("compute_forces")) {
        std::cout << "Has compute_forces attribute." << std::endl;
    } else {
        std::cout << "No compute_forces attribute." << std::endl;
    }
    particles.neighbor_list->list_used = false;
}

double MLIAP::potential(const particles_instance& particles) {
    try
    {
        auto dtype = torch::kFloat32;
        torch::Tensor coordinates_cpu = torch::empty({N_at, 3}, torch::TensorOptions().dtype(dtype).device(torch::kCPU));
        auto accessor = coordinates_cpu.accessor<float, 2>();

        for (int i = 0; i < N_at; ++i) {
            for (int j = 0; j < 3; ++j) {
                accessor[i][j] = static_cast<float>(particles.h_x(i, j));
            }
        }

        torch::Tensor coordinates = coordinates_cpu.to(device_);
        torch::Tensor species = labels_to_species(particles.label_xyz, device_);

        // Edge index shape: (2, N_edge)
        std::vector<int64_t> row_src_vec;
        std::vector<int64_t> row_dst_vec;
        //if (!particles.neighbor_list->list_used)
        //{
            auto& verlet_list = particles.neighbor_list->h_verlet_list;
            Kokkos::deep_copy(verlet_list,particles.neighbor_list->verlet_list);
            auto& neighbour_count = particles.neighbor_list->h_neighbour_count;
            Kokkos::deep_copy(neighbour_count,particles.neighbor_list->neighbour_count);

            for (int64_t i = 0; i < N_at; ++i) {
                int64_t num_neighbors = neighbour_count(i);
                for (int64_t j = 0; j < num_neighbors; ++j) {
                    int64_t neighbor = verlet_list(i, j);
                    row_src_vec.push_back(i);
                    row_dst_vec.push_back(neighbor);
                    // duplicate edges
                    row_src_vec.push_back(neighbor);
                    row_dst_vec.push_back(i);
                }
            }
            particles.neighbor_list->list_used = true;
        //}
        int64_t N_edge = static_cast<int64_t>(row_src_vec.size());
        torch::Tensor edge_index = torch::stack({
                torch::from_blob(row_src_vec.data(), {N_edge}, torch::TensorOptions().dtype(torch::kInt64)).clone(),
                torch::from_blob(row_dst_vec.data(), {N_edge}, torch::TensorOptions().dtype(torch::kInt64)).clone()
            }, 0).to(device_);
        // Batch shape: (N_at,) all zeros for single structure
        torch::Tensor batch = torch::zeros({N_at}, torch::TensorOptions().dtype(torch::kInt64).device(device_));


        // No periodic boundary conditions -> shifts=None, cell=None
        // In TorchScript, optional IValues are represented by c10::optional<torch::Tensor> (or just omit).
        // We'll push IValue() (None) for these.
        c10::IValue shifts = c10::IValue();
        //c10::IValue cell = c10::IValue();
        c10::IValue atom_mask = c10::IValue();
        c10::IValue dataset_index = c10::IValue();

        const float Lx = particles.L[0];
        const float Ly = particles.L[1];
        const float Lz = particles.L[2];

        // Set cell for periodic boundary conditions
        auto cell_cpu = torch::tensor(
            {{{Lx, 0.0f, 0.0f},
            {0.0f, Ly, 0.0f},
            {0.0f, 0.0f, Lz}}},
            torch::TensorOptions().dtype(torch::kFloat32));
        torch::Tensor cell_t = cell_cpu.to(device_);
        c10::IValue cell = cell_t;

        std::vector<torch::jit::IValue> inputs;
        inputs.reserve(8);
        inputs.push_back(coordinates);
        inputs.push_back(species);
        inputs.push_back(edge_index);
        inputs.push_back(batch);
        inputs.push_back(shifts);        // None
        inputs.push_back(cell);          // None
        inputs.push_back(atom_mask);     // None
        inputs.push_back(dataset_index); // None

        // Forward pass
        auto out_ivalue = module_.forward(inputs);

        if (!out_ivalue.isGenericDict()) {
            throw std::runtime_error("Expected forward to return a dict.");
        }

        auto out_dict = out_ivalue.toGenericDict();

        // Retrieve outputs
        torch::Tensor energy;
        double V = 0.0;
        if (out_dict.contains("energy")) {
            auto t = out_dict.at("energy").toTensor().to(torch::kCPU);  // or .cpu()
            V = t.item<double>();
        }
        return V*evtointernal;
    } catch (const c10::Error& e) {
        std::cerr << "TorchScript error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;}
    return 0.0;
}

void tensor_forces_to_kokkos(const torch::Tensor& forces_in, type_f& f) {
    torch::Tensor t = forces_in.to(torch::kCPU);

    // Convert to double and make contiguous
    if (t.dtype() != torch::kFloat64) t = t.to(torch::kFloat64);
    if (!t.is_contiguous()) t = t.contiguous();

    const double* src = t.data_ptr<double>();

    // copy values to force view
    for (int64_t i = 0; i < f.extent(0); ++i) {
        f(i, 0) = - src[i * 3 + 0] * evtointernal;
        f(i, 1) = - src[i * 3 + 1] * evtointernal;
        f(i, 2) = - src[i * 3 + 2] * evtointernal;
    }
}

void MLIAP::force(const particles_instance& particles, type_f& f) {
    try
    {
        auto dtype = torch::kFloat32;
        // Coordinates shape: (N_at, 3)
        
        torch::Tensor coordinates_cpu = torch::empty({N_at,3}, torch::TensorOptions().dtype(dtype).device(torch::kCPU));
        {
            auto acc = coordinates_cpu.accessor<float,2>();
            for (int i=0;i<N_at;++i)
                for (int j=0;j<3;++j)
                    acc[i][j] = static_cast<float>(particles.h_x(i,j));
        }
        torch::Tensor coordinates = coordinates_cpu.to(device_);

        // Species shape: (N_at,)  (atomic numbers, e.g., 1=H, 6=C, 8=O)
        // torch::Tensor species = torch::tensor({1, 1, 8}, torch::TensorOptions().dtype(torch::kInt64)).to(device);
        torch::Tensor species = labels_to_species(particles.label_xyz, device_);

        // Edge index shape: (2, N_edge).
        std::vector<int64_t> row_src_vec;
        std::vector<int64_t> row_dst_vec;
        //if (!particles.neighbor_list->list_used)
        //{
            auto& verlet_list = particles.neighbor_list->h_verlet_list;
            Kokkos::deep_copy(verlet_list,particles.neighbor_list->verlet_list);
            auto& neighbour_count = particles.neighbor_list->h_neighbour_count;
            Kokkos::deep_copy(neighbour_count,particles.neighbor_list->neighbour_count);

            for (int64_t i = 0; i < N_at; ++i) {
                int64_t num_neighbors = neighbour_count(i);
                for (int64_t j = 0; j < num_neighbors; ++j) {
                    int64_t neighbor = verlet_list(i, j);
                    row_src_vec.push_back(i);
                    row_dst_vec.push_back(neighbor);
                    // duplicate edges
                    row_src_vec.push_back(neighbor);
                    row_dst_vec.push_back(i);
                }
            }
            particles.neighbor_list->list_used = true;
        //}
        int64_t N_edge = static_cast<int64_t>(row_src_vec.size());
        torch::Tensor edge_index = torch::stack({
                torch::from_blob(row_src_vec.data(), {N_edge}, torch::TensorOptions().dtype(torch::kInt64)).clone(),
                torch::from_blob(row_dst_vec.data(), {N_edge}, torch::TensorOptions().dtype(torch::kInt64)).clone()
            }, 0).to(device_);

        // Batch shape: (N_at,) all zeros for single structure
        torch::Tensor batch = torch::zeros({N_at}, torch::TensorOptions().dtype(torch::kInt64).device(device_));

        // No periodic boundary conditions -> shifts=None, cell=None
        // In TorchScript, optional IValues are represented by c10::optional<torch::Tensor> (or just omit).
        // We'll push IValue() (None) for these.
        c10::IValue shifts = c10::IValue();
        //c10::IValue cell = c10::IValue();
        c10::IValue atom_mask = c10::IValue();
        c10::IValue dataset_index = c10::IValue();

        const float Lx = particles.L[0];
        const float Ly = particles.L[1];
        const float Lz = particles.L[2];

        // Set cell for periodic boundary conditions
        auto cell_cpu = torch::tensor(
            {{{Lx, 0.0f, 0.0f},
            {0.0f, Ly, 0.0f},
            {0.0f, 0.0f, Lz}}},
            torch::TensorOptions().dtype(torch::kFloat32));
        torch::Tensor cell_t = cell_cpu.to(device_);
        c10::IValue cell = cell_t;

        std::vector<torch::jit::IValue> inputs;
        inputs.reserve(8);
        inputs.push_back(coordinates);
        inputs.push_back(species);
        inputs.push_back(edge_index);
        inputs.push_back(batch);
        inputs.push_back(shifts);        // None
        inputs.push_back(cell);          // None
        inputs.push_back(atom_mask);     // None
        inputs.push_back(dataset_index); // None
        // Forward pass
        auto out_ivalue = module_.forward(inputs);

        if (!out_ivalue.isGenericDict()) {
            throw std::runtime_error("Expected forward to return a dict.");
        }

        auto out_dict = out_ivalue.toGenericDict();
        torch::Tensor forces;
        forces = out_dict.at("forces").toTensor().to(torch::kCPU);
        tensor_forces_to_kokkos(forces, f);
    } catch (const c10::Error& e) {
        std::cerr << "TorchScript error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
}