#include "DeePMD.hpp"
#include <map>
#include <sstream>
#include <stdexcept>

DeePMD::DeePMD(const YAML::Node& config)
{
    name = "DeePMD";
    if (!config["model_file"]) {
        throw std::runtime_error("Cannot initialize DeePMD: missing model_file.");
    }
    model_file = config["model_file"].as<std::string>();
    if (config["gpu_rank"]) gpu_rank = config["gpu_rank"].as<int>();
}

void DeePMD::init(const particles_instance& particles)
{
    dp.init(model_file, gpu_rank);

    x_deepmd = Kokkos::View<double**, Kokkos::HostSpace>("x_deepmd", particles.N, 3);
    h_f = Kokkos::View<double**, Kokkos::HostSpace>("h_f", particles.N, 3);
    coordinates.resize(3 * particles.N);
    atom_types.resize(particles.N);
    box.resize(9);
    forces.resize(3 * particles.N);
    virial.resize(9);

    std::string model_type_map;
    dp.get_type_map(model_type_map);
    std::istringstream type_map_stream(model_type_map);
    std::map<std::string, int> type_map;
    std::string label;
    int type_idx = 0;
    while (type_map_stream >> label) {
        type_map[label] = type_idx++;
    }
    if (type_map.empty()) {
        throw std::runtime_error("Cannot initialize DeePMD: model has no species type map.");
    }

    for (int i = 0; i < particles.N; ++i) {
        label = particles.h_atom_type_list(particles.h_id(i)).label;
        auto it = type_map.find(label);
        if (it == type_map.end()) {
            throw std::runtime_error("Cannot initialize DeePMD: unsupported atom type " + label + ".");
        }
        atom_types[i] = it->second;
    }
}

double DeePMD::evaluate(const particles_instance& particles)
{
    Kokkos::deep_copy(x_deepmd, particles.x);
    for (int i = 0; i < particles.N; ++i) {
        for (int d = 0; d < 3; ++d) {
            coordinates[3 * i + d] = x_deepmd(i, d);
        }
    }
    box = {particles.L[0],0.0,0.0,
                         0.0,particles.L[1],0.0,
                         0.0,0.0,particles.L[2]};

    double energy;
    dp.compute(energy, forces, virial, coordinates, atom_types, box);
    return energy;
}

double DeePMD::potential(const particles_instance& particles) {
    Kokkos::Timer deepmd_timer;
    double energy = evaluate(particles);
    time_potential += deepmd_timer.seconds();
    return energy*evtointernal;
}

void DeePMD::force(const particles_instance& particles, type_f& f) {
    Kokkos::Timer deepmd_timer;
    evaluate(particles);
    Kokkos::deep_copy(h_f, f);

    for (int i = 0; i < particles.N; ++i) {
        for (int d = 0; d < 3; ++d) {
            h_f(i, d) -= forces[3 * i + d]*evtointernal;
        }
    }

    Kokkos::deep_copy(f, h_f);
    time_force += deepmd_timer.seconds();
}