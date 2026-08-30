#include "MBX.hpp"

MBX::MBX(const YAML::Node& config)
{
    // defaults to "mbx.json" if nothing is supplied by user
    if (config["json_file"]) json_filename = config["json_file"].as<std::string>();
}

void MBX::init(const particles_instance& particles) 
{
    x_mbx = Kokkos::View<double**, Kokkos::HostSpace>("x_mbx", particles.N, 3);
    h_f = Kokkos::View<double**, Kokkos::HostSpace>("h_f", particles.N, 3);

    Kokkos::deep_copy(x_mbx, particles.x);
    build_molecule_map(particles);

    // The json (settings) and nrg (monomers and coordinates) files
    std::string json = json_filename;

    // Convert the strings to char
    char json_c[json.size() + 1];
    std::strcpy(json_c,json.c_str());

    // Write energy file
    std::vector<double> coordinates = coordinates_to_vector(particles);

    // Read the file and setup the system
    systems.clear();
    systems.emplace_back();
    for (std::size_t i = 0; i < water_molecules.size(); ++i) {
        std::vector<double> xyz(coordinates.begin() + 9 * i, coordinates.begin() + 9 * (i + 1));
        std::vector<std::string> atom_names = {"O", "H", "H"};
        systems[0].AddMonomer(xyz, atom_names, "h2o");
    }
    systems[0].Initialize();
    systems[0].SetUpFromJson(json_c);
    box_initialized = false;

    // Set boxsize
    update_box(particles);
}

double MBX::potential(const particles_instance& particles) {
    Kokkos::deep_copy(x_mbx, particles.x);

    // Update the coords for MBX first
    systems[0].SetRealXyz(coordinates_to_vector(particles));
    update_box(particles);
    return systems[0].Energy(false)*kcaltointernal;
}

void MBX::force(const particles_instance& particles, type_f& f) {
    Kokkos::deep_copy(x_mbx, particles.x);

    systems[0].SetRealXyz(coordinates_to_vector(particles));
    update_box(particles);

    // Compute energy and gradients.
    // The true argument tells MBX to compute gradients.
    systems[0].Energy(true);

    std::vector<double> grads = systems[0].GetRealGrads();

    Kokkos::deep_copy(h_f, f);

    for (std::size_t i_mbx = 0; i_mbx < particles.N; ++i_mbx) {
        std::size_t i = mbx_to_particle[i_mbx];

        h_f(i, 0) += grads[3 * i_mbx + 0]*kcaltointernal;
        h_f(i, 1) += grads[3 * i_mbx + 1]*kcaltointernal;
        h_f(i, 2) += grads[3 * i_mbx + 2]*kcaltointernal;
    }

    Kokkos::deep_copy(f, h_f);
}

void MBX::build_molecule_map(const particles_instance& particles)
{
    using atom_index_t = std::size_t;

    water_molecules.clear();
    mbx_to_particle.clear();

    std::map<int, std::vector<atom_index_t>> molecules;

    for (atom_index_t i = 0; i < particles.N; ++i) {
        int mol_id = particles.h_mol_id(i);
        molecules[mol_id].push_back(i);
    }

    if (molecules.empty()) {
        throw std::runtime_error("Cannot initialize MBX: no molecules found.");
    }

    for (std::map<int, std::vector<atom_index_t>>::const_iterator it = molecules.begin(); it != molecules.end(); ++it) {
        int mol_id = it->first;
        const std::vector<atom_index_t>& atom_indices = it->second;
        atom_index_t oxygen = particles.N;
        std::vector<atom_index_t> hydrogens;

        for (atom_index_t i : atom_indices) {
            int type_idx = particles.h_id(i);
            std::string label = particles.h_atom_type_list(type_idx).label;

            if (label == "O") {
                if (oxygen != static_cast<atom_index_t>(particles.N)) {
                    throw std::runtime_error("Molecule id " + std::to_string(mol_id) + " has more than one oxygen atom.");
                }
                oxygen = i;
            } else if (label == "H") {
                hydrogens.push_back(i);
            } else {
                throw std::runtime_error("Molecule id " + std::to_string(mol_id) + " contains unsupported atom type " + label + ".");
            }
        }

        if (oxygen == static_cast<atom_index_t>(particles.N) || hydrogens.size() != 2 || atom_indices.size() != 3) {
            throw std::runtime_error("Molecule id " + std::to_string(mol_id) + " is not a valid H2O molecule.");
        }

        WaterMolecule molecule = {oxygen, hydrogens[0], hydrogens[1]};
        water_molecules.push_back(molecule);
        mbx_to_particle.push_back(molecule.oxygen);
        mbx_to_particle.push_back(molecule.hydrogen1);
        mbx_to_particle.push_back(molecule.hydrogen2);
    }
}

void MBX::update_box(const particles_instance& particles)
{
    if (box_initialized &&
        box_length[0] == particles.L[0] &&
        box_length[1] == particles.L[1] &&
        box_length[2] == particles.L[2]) return;

    std::vector<double> box = {particles.L[0],0.0,0.0,
                                0.0,particles.L[1],0.0,
                                0.0,0.0,particles.L[2]};
    systems[0].SetPBC(box);
    box_length[0] = particles.L[0];
    box_length[1] = particles.L[1];
    box_length[2] = particles.L[2];
    box_initialized = true;
}

void MBX::write_nrg_file(const particles_instance& particles, const std::string& filename)
{
    // Writes the .nrg file from internal data
    using atom_index_t = std::size_t;

    mbx_to_particle.clear();

    std::map<int, std::vector<atom_index_t>> molecules;

    for (atom_index_t i = 0; i < particles.N; ++i) {
        int mol_id = particles.h_mol_id(i);
        molecules[mol_id].push_back(i);
    }

    if (molecules.empty()) {
        throw std::runtime_error("Cannot write NRG file: no molecules found.");
    }

    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Could not open NRG file for writing: " + filename);
    }

    out << "SYSTEM " << molecules.size() << "H2O\n";

    for (std::map<int, std::vector<atom_index_t>>::const_iterator it = molecules.begin(); it != molecules.end(); ++it) {
        int mol_id = it->first;
        std::vector<atom_index_t> atom_indices = it->second;

        // Sort atoms so O comes first, then H, H.
        // mass < 6  -> H // arbitrary value of 6 since we only handle water here anyways
        // mass >= 6 -> O
        std::stable_sort(atom_indices.begin(), atom_indices.end(),
            [&](atom_index_t a, atom_index_t b) {
                int type_a = particles.h_id(a);
                int type_b = particles.h_id(b);

                double mass_a = particles.h_atom_type_list(type_a).mass;
                double mass_b = particles.h_atom_type_list(type_b).mass;

                int order_a = (mass_a < 6.0) ? 1 : 0;
                int order_b = (mass_b < 6.0) ? 1 : 0;

                return order_a < order_b;
            });

        int n_o = 0;
        int n_h = 0;

        for (atom_index_t i : atom_indices) {
            int type_idx = particles.h_id(i);
            double mass = particles.h_atom_type_list(type_idx).mass;

            if (mass < 6.0) {
                ++n_h;
            } else {
                ++n_o;
            }
        }

        if (n_o != 1 || n_h != 2 || atom_indices.size() != 3) {
            throw std::runtime_error(
                "Molecule id " + std::to_string(mol_id) +
                " is not a valid H2O molecule. Found " +
                std::to_string(n_o) + " O atoms, " +
                std::to_string(n_h) + " H atoms, " +
                std::to_string(atom_indices.size()) + " total atoms."
            );
        }

        out << "MOLECULE\n";
        out << "MONOMER h2o\n";

        for (atom_index_t i : atom_indices) {
            mbx_to_particle.push_back(i);

            int type_idx = particles.h_id(i);
            double mass = particles.h_atom_type_list(type_idx).mass;

            const char* label = (mass < 6.0) ? "H" : "O";

            out << " "
                << std::left << std::setw(2) << label
                << std::right << std::fixed << std::setprecision(8)
                << std::setw(18) << x_mbx(i, 0)
                << std::setw(14) << x_mbx(i, 1)
                << std::setw(14) << x_mbx(i, 2)
                << "\n";
        }

        out << "ENDMON\n";
        out << "ENDMOL\n";
    }

    out << "ENDSYS\n";
}

std::vector<double> MBX::coordinates_to_vector(const particles_instance& particles) const
{
    std::vector<double> xyz(3 * particles.N);

    for (std::size_t i = 0; i < water_molecules.size(); ++i) {
        const WaterMolecule& molecule = water_molecules[i];
        std::size_t atom_indices[3] = {molecule.oxygen, molecule.hydrogen1, molecule.hydrogen2};

        for (std::size_t dir = 0; dir < 3; ++dir) {
            double oxygen_coordinate = x_mbx(molecule.oxygen, dir);
            xyz[9 * i + dir] = oxygen_coordinate;

            for (std::size_t atom = 1; atom < 3; ++atom) {
                double displacement = x_mbx(atom_indices[atom], dir) - oxygen_coordinate;
                displacement -= std::round(displacement / particles.L[dir]) * particles.L[dir];
                xyz[9 * i + 3 * atom + dir] = oxygen_coordinate + displacement;
            }
        }
    }

    return xyz;
}