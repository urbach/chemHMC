#include "Input_reader.hpp"
#include "Parameters.hpp"
#include "Calc_Manager.hpp"
#include "Calc.hpp"
#include "bonds.hpp"
#include "../modules/neighbor_list/Neighbor_list.hpp"
#include "potentials/non_bonded_interactions/LJ.hpp"
#include "atom.hpp"
#include "potentials/non_bonded_interactions/coulomb_ewald.hpp"
#include "potentials/non_bonded_interactions/coulomb_direct.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <iostream>
#include <array>
#include <algorithm>

Input_reader::Input_reader(params_class* params, integrator_type*& integrator,particles_instance*& particles, Calc_Manager*& calc_manager) 
                : params_ptr(params), integrator_ptr(integrator), particles_ptr(particles), calc_manager_ptr(calc_manager) {
    if (!params) {
        throw std::runtime_error("Error: Null pointer passed as params to Input_reader constructor.");
    }
}

void Input_reader::validate_file(const std::string& file_path) const {
    std::ifstream file(file_path);
    if (!file.good()) {
        throw std::runtime_error("Error: Unable to open file: " + file_path);
    }
}

template <typename T>
T check_and_assign_value(const YAML::Node& node, const char* tag) {
    if (!node[tag]) {
        throw std::runtime_error(std::string("Error: Missing tag in YAML: ") + tag);
    }
    try {
        return node[tag].as<T>();
    } catch (const YAML::TypedBadConversion<T>& e) {
        throw std::runtime_error(std::string("Error: Type mismatch for tag: ") + tag);
    }
}

void Input_reader::parse_input(int argc, char** argv) {
    int opt = -1;
    // search for command line option and put filename in "infilename"
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-i") == 0) {
            opt = i + 1;
            break;
        }
    }
    if (opt < 0 || opt == argc) {
        std::cout << "No input file specified, Aborting" << std::endl;
        std::cout << "usage:  ./main -i infile.in" << std::endl;
        Kokkos::abort("");
    }
    std::string input_file = argv[opt];
    validate_file(input_file);
    try {
        doc = YAML::LoadFile(input_file);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Error parsing YAML file: " + std::string(e.what()));
    }

    parse_simulation_parameters(doc);
    parse_integrator_options(doc);
    get_number_of_particles();
    parse_particles_options(doc);
    particles_ptr->InitX();
    assign_ids();
    read_xyz();
    populate_calc_list(doc);
}

void Input_reader::parse_simulation_parameters(YAML::Node& doc) {
    // Get parameters
    params_ptr->L[0] = check_and_assign_value<double>(doc["geometry"], "Lx");
    params_ptr->L[1] = check_and_assign_value<double>(doc["geometry"], "Ly");
    params_ptr->L[2] = check_and_assign_value<double>(doc["geometry"], "Lz");
    params_ptr->inverse_L[0] = 1.0/params_ptr->L[0];
    params_ptr->inverse_L[1] = 1.0/params_ptr->L[1];
    params_ptr->inverse_L[2] = 1.0/params_ptr->L[2];
    params_ptr->inverse_halved_L[0] = 2.0*params_ptr->inverse_L[0];
    params_ptr->inverse_halved_L[1] = 2.0*params_ptr->inverse_L[1];
    params_ptr->inverse_halved_L[2] = 2.0*params_ptr->inverse_L[2];
    params_ptr->seed = check_and_assign_value<int>(doc, "seed");
    params_ptr->nameout = check_and_assign_value<std::string>(doc, "output_file");
    params_ptr->parameter_file = check_and_assign_value<std::string>(doc, "parameter_file");
    params_ptr->Ntrajectories = check_and_assign_value<int>(doc, "Ntrajectories");
    params_ptr->thermalization_steps = check_and_assign_value<int>(doc, "thermalization_steps");
    params_ptr->save_every = check_and_assign_value<int>(doc, "save_every");
    params_ptr->print_info_every = check_and_assign_value<int>(doc, "print_info_every");
    params_ptr->seed = check_and_assign_value<int>(doc, "seed");

    if (doc["start_configuration_file"]) 
    {   // positions may also be read from lammps datafile
        params_ptr->start_configuration_file = check_and_assign_value<std::string>(doc, "start_configuration_file");
    } else {
        Kokkos::abort("Since x is initialized via the start_conf_file, one has to be provided for now");
    }
    std::string simulation_type = check_and_assign_value<std::string>(doc, "simulation_type");
    if (simulation_type == "MD") MD = true;

    // Read number of particles from the starting configuration
    std::ifstream xyz_file(params_ptr->nameout);
    std::string line;
    if (std::getline(xyz_file, line)) {
        params_ptr->N = std::atoi(line.c_str());
        std::cout << "Number of particles (N) read from .xyz file: " << params_ptr->N << std::endl;
    } else {
        std::cerr << "Error: Could not read the first line of file " << params_ptr->nameout << std::endl;
        throw std::runtime_error("File read error");
    }
    xyz_file.close();
    // Check output file
    params_ptr->fileout = fopen(params_ptr->nameout.c_str(), "ab");
    if (params_ptr->fileout == NULL || params_ptr->nameout.length() <= 0 || params_ptr->nameout.compare("null") == 0) {
        printf("unable to open file %s\n", params_ptr->nameout.c_str());
        Kokkos::abort("exiting...");
    }
}

void Input_reader::parse_integrator_options(YAML::Node& doc) {
    // Set integrator type
    if (doc["integrator"]) {
        std::string name = check_and_assign_value<std::string>(doc["integrator"], "name");
        if (name == "LEAP") {
            integrator_ptr = new LEAP(doc, *params_ptr);
        } else if (name == "OMF2") {
            integrator_ptr = new OMF2(doc, *params_ptr);
        } else if (name == "OMF4") {
            integrator_ptr = new OMF4(doc, *params_ptr);
        } else if (name == "VELOCITY_VERLET") {
            integrator_ptr = new VELOCITY_VERLET(doc, *params_ptr);
        } else if (name == "VELOCITY_VERLET_SHAKE") {
            integrator_ptr = new VELOCITY_VERLET_SHAKE(doc, *params_ptr);
        } else {
            throw std::runtime_error("Error: Invalid integrator name: " + name);
        }
    } else {
        throw std::runtime_error("Error: No integrator specified in input file.");
    }
    integrator_ptr->dt = check_and_assign_value<double>(doc["integrator"], "dt");
    integrator_ptr->steps = check_and_assign_value<int>(doc["integrator"], "steps");
    integrator_ptr->particles = particles_ptr;
}

void get_atom_types_from_file(YAML::Node& doc, particles_instance*& particles) {
    std::string parameter_file = check_and_assign_value<std::string>(doc, "parameter_file");

    std::ifstream infile(parameter_file);
    std::string line;
    std::getline(infile, line); // Skip the first line
    std::vector<atom_type> temp_atom_type_list;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string label;
        double mass, charge, epsilon, sigma;
        int index;

        iss >> index >> label >> mass >> charge >> epsilon >> sigma;

        // Create an atom_type instance and add it to the vector
        atom_type atom(label.c_str(), mass, charge, index, epsilon*kcaltointernal, sigma);
        temp_atom_type_list.push_back(atom);
    }
    particles->h_atom_type_list = Kokkos::create_mirror_view(Kokkos::View<atom_type*>("atom_type_list", temp_atom_type_list.size()));
    for (size_t i = 0; i < temp_atom_type_list.size(); ++i) {
        particles->h_atom_type_list(i) = temp_atom_type_list[i];
    }
    particles->atom_type_list = Kokkos::View<atom_type*>("atom_type_list", temp_atom_type_list.size());

    Kokkos::deep_copy(particles->atom_type_list, particles->h_atom_type_list);
}

void mix_pair_parameters(YAML::Node& doc, particles_instance*& particles) {

    // assign correct size to parameter matrices
    int num_atom_types = particles->atom_type_list.extent(0);
    particles->sigma_mat = Kokkos::View<double**>("sigma_mat", num_atom_types, num_atom_types);
    particles->epsilon_mat = Kokkos::View<double**>("epsilon_mat", num_atom_types, num_atom_types);

    // Initialize host mirrors
    particles->h_sigma_mat = Kokkos::create_mirror_view(particles->sigma_mat);
    particles->h_epsilon_mat = Kokkos::create_mirror_view(particles->epsilon_mat);


    // Fill known diagonal elements
    for(int i = 0; i < particles->h_atom_type_list.extent(0); ++i) {
        particles->h_epsilon_mat(i, i) = particles->h_atom_type_list(i).LJ_epsilon;
        particles->h_sigma_mat(i, i) = particles->h_atom_type_list(i).LJ_sigma;
    }

    // Apply Lorentz-Berthelot mixing rules
    for(int i = 0; i < particles->h_atom_type_list.extent(0); ++i) {
        for(int j = i + 1; j < particles->h_atom_type_list.extent(0); ++j) {
            double epsilon = sqrt(particles->h_epsilon_mat(i, i) * particles->h_epsilon_mat(j, j)); // Berthelots rule
            double sigma = 0.5 * (particles->h_sigma_mat(i, i) + particles->h_sigma_mat(j, j)); // Lorentz rule
            particles->h_epsilon_mat(i, j) = epsilon;
            particles->h_epsilon_mat(j, i) = epsilon;
            particles->h_sigma_mat(i, j) = sigma;
            particles->h_sigma_mat(j, i) = sigma;
            //printf("i: %d j: %d eps: %f sig: %f",i,j,epsilon,sigma);
        }

    }

    // Copy the updated data to device memory
    Kokkos::deep_copy(particles->sigma_mat, particles->h_sigma_mat);
    Kokkos::deep_copy(particles->epsilon_mat, particles->h_epsilon_mat);
}

void Input_reader::parse_particles_options(YAML::Node& doc) {
    // get interaction parameters from parameter file
    get_atom_types_from_file(doc, particles_ptr);
    // generate mixed pair parameters
    mix_pair_parameters(doc, particles_ptr);

    // get the Rest of the parameters
    particles_ptr->T = check_and_assign_value<double>(doc["particles"], "temperature");
    particles_ptr->beta = 1/(kB*particles_ptr->T);
    particles_ptr->sbeta = sqrt(particles_ptr->beta);
    particles_ptr->start_configuration_file = check_and_assign_value<std::string>(doc, "start_configuration_file");

    particles_ptr->L[0] = params_ptr->L[0];
    particles_ptr->L[1] = params_ptr->L[1];
    particles_ptr->L[2] = params_ptr->L[2];

    particles_ptr->inverse_L[0] = 1.0/params_ptr->L[0];
    particles_ptr->inverse_L[1] = 1.0/params_ptr->L[1];
    particles_ptr->inverse_L[2] = 1.0/params_ptr->L[2];
    
    particles_ptr->inverse_halved_L[0] = 2.0*params_ptr->inverse_L[0];
    particles_ptr->inverse_halved_L[1] = 2.0*params_ptr->inverse_L[1];
    particles_ptr->inverse_halved_L[2] = 2.0*params_ptr->inverse_L[2];
    
    particles_ptr->compute_coeff_momenta();
    particles_ptr->compute_coeff_position();
    particles_ptr->rand_pool.init(params_ptr->seed, particles_ptr->N);
}

void Input_reader::get_number_of_particles() {
    if (doc["lammps_data_file"]) {
        std::ifstream infile(doc["lammps_data_file"].as<std::string>());
        if (!infile) {
            throw std::runtime_error("Unable to open LAMMPS data file: " + doc["lammps_data_file"].as<std::string>());
        }
        std::string line;
        while (std::getline(infile, line)) {
            std::istringstream iss(line);
            int count;
            std::string keyword;
            if (iss >> count) {
                std::getline(iss, keyword);
                keyword = keyword.substr(keyword.find_first_not_of(" \t"));
                if (keyword == "atoms") {
                    particles_ptr->N = count;
                    return;
                }
            }
        }
        throw std::runtime_error("Number of atoms not found in LAMMPS data file.");
    }
    std::ifstream infile(params_ptr->start_configuration_file);
    if (!infile) {
        throw std::runtime_error("Unable to open parameter file: " + params_ptr->start_configuration_file);
    }

    std::string line;
    // get number of particles from first line
    std::getline(infile, line);
    std::istringstream iss(line);
    iss >> particles_ptr->N;
}

void Input_reader::assign_ids() {
    // this function reads in the atom types from the start_configuration_file
    // and assigns each an atom_type_id defined in the atom_type_list.
    std::ifstream infile(params_ptr->start_configuration_file);
    if (!infile) {
        throw std::runtime_error("Unable to open parameter file: " + params_ptr->start_configuration_file);
    }

    std::string line;
    // get number of particles from first line
    std::getline(infile, line);
    std::istringstream iss(line);
    int N_particles;
    iss >> N_particles;

    // skip comment line 
    std::getline(infile, line);
    // assign all ids
    for (int i = 0; i < N_particles; i++) {
        // get type from current line
        std::getline(infile, line);
        std::istringstream iss(line);
        std::string type;
        iss >> type;
        
        // find matching id and assign it
        for (int j = 0; j < particles_ptr->h_atom_type_list.extent(0); j++) {
            if(type == particles_ptr->h_atom_type_list[j].label) {
                // assign id and shift by -1 so ids allign with indices of parameter views
                particles_ptr->h_id[i] = particles_ptr->h_atom_type_list[j].type_index-1;
                break;
            }
        }
    }
    Kokkos::deep_copy(particles_ptr->id, particles_ptr->h_id);
    Kokkos::fence();
}

void Input_reader::read_xyz() {
    // if a lammps data file is available, we read atoms positions from that
    if (doc["lammps_data_file"]) {
        std::ifstream infile(doc["lammps_data_file"].as<std::string>());
        if (!infile.is_open()) {
            std::cerr << "Error opening LAMMPS data file: " << doc["lammps_data_file"].as<std::string>() << std::endl;
            Kokkos::abort("abort");
        }

        std::string line;
        bool inAtomSection = false;
        while (std::getline(infile, line)) {
            if (line.find("Atoms") != std::string::npos) {
                inAtomSection = true;
                break;
            }
        }

        if (!inAtomSection) {
            std::cerr << "Error: 'Atoms' section not found in LAMMPS data file" << std::endl;
            Kokkos::abort("abort");
        }

        // Skip one line after section title
        std::getline(infile, line);

        for (int i = 0; i < particles_ptr->N; ++i) {
            if (!std::getline(infile, line)) {
                std::cerr << "Error: unexpected end of file while reading atom positions" << std::endl;
                Kokkos::abort("abort");
            }
            std::istringstream iss(line);
            int id, type, molecule_id;
            double x, y, z, charge;
            if (!(iss >> id >> molecule_id >> type >> charge >> x >> y >> z)) {
                std::cerr << "Error parsing atom position at line " << i + 1 << std::endl;
                Kokkos::abort("abort");
            }
            id -= 1; // zero-based indexing
            particles_ptr->h_x(id, 0) = x;
            particles_ptr->h_x(id, 1) = y;
            particles_ptr->h_x(id, 2) = z;
        }


        infile.close();
        Kokkos::deep_copy(particles_ptr->x, particles_ptr->h_x);
        return;
    }
    // otherwise we just read it from the configuration file
    // Open the input file using ifstream
    std::ifstream infile(params_ptr->start_configuration_file);
    if (!infile.is_open()) {
        std::cerr << "Error opening file " << params_ptr->start_configuration_file << std::endl;
        Kokkos::abort("abort");
    }

    // Count the number of lines in the file
    int lines = 0;
    std::string temp_line;
    while (std::getline(infile, temp_line)) {
        lines++;
    }

    // Check if the number of lines is a multiple of N + 2
    if (lines % (particles_ptr->N + 2) != 0) {
        std::cerr << "Error: input file " << params_ptr->start_configuration_file << " contains " << lines << " lines" << std::endl;
        std::cerr << "       the number of lines must be a multiple of N+2 = " << particles_ptr->N + 2 << std::endl;
        Kokkos::abort("abort");
    }

    // Calculate the number of configurations in the file
    int confs = lines / (particles_ptr->N + 2);
    std::cout << "Number of configurations in input file: " << confs << std::endl;

    // Reset the file stream to the beginning
    infile.clear();
    infile.seekg(0, std::ios::beg);

    // Skip lines to reach the last configuration
    int lines_to_skip = (confs - 1) * (particles_ptr->N + 2);
    for (int i = 0; i < lines_to_skip; ++i) {
        if (!std::getline(infile, temp_line)) {
            std::cerr << "Error: unexpected end of file while skipping to last configuration" << std::endl;
            Kokkos::abort("abort");
        }
    }

    // Read the number of atoms from the first line of the last configuration
    if (!std::getline(infile, temp_line)) {
        std::cerr << "Error: unexpected end of file while reading number of atoms" << std::endl;
        Kokkos::abort("abort");
    }
    int num_atoms_in_file = std::stoi(temp_line);
    if (num_atoms_in_file != particles_ptr->N) {
        std::cerr << "Error: number of atoms in file (" << num_atoms_in_file << ") does not match expected N (" << particles_ptr->N << ")" << std::endl;
        Kokkos::abort("abort");
    }

    // Read the comment line (we can skip or store it if needed)
    if (!std::getline(infile, temp_line)) {
        std::cerr << "Error: unexpected end of file while reading comment line" << std::endl;
        Kokkos::abort("abort");
    }
    // Optionally, store or skip the comment line
    std::string comment_line = temp_line;

    std::cout << "Reading last configuration from input file " << params_ptr->start_configuration_file << std::endl;

    // Read the atom data
    particles_ptr->label_xyz.clear(); // Ensure label_xyz is empty before filling
    for (int i = 0; i < particles_ptr->N; ++i) {
        if (!std::getline(infile, temp_line)) {
            std::cerr << "Error: unexpected end of file while reading atom data" << std::endl;
            Kokkos::abort("abort");
        }
        std::istringstream iss(temp_line);
        std::string id;
        double x_val, y_val, z_val;
        if (!(iss >> id >> x_val >> y_val >> z_val)) {
            std::cerr << "Error parsing atom data on line " << i + 1 << std::endl;
            Kokkos::abort("Error parsing xyz file");
        }
        particles_ptr->label_xyz.push_back(id);
        particles_ptr->h_x(i, 0) = x_val;
        particles_ptr->h_x(i, 1) = y_val;
        particles_ptr->h_x(i, 2) = z_val;
    }

    infile.close();
    Kokkos::deep_copy(particles_ptr->x, particles_ptr->h_x);
}

void Input_reader::populate_calc_list(YAML::Node& doc) {
    if (doc["LJ"]) {
        std::string algorithm = check_and_assign_value<std::string>(doc["LJ"], "algorithm");
        if (algorithm.compare("cutoff") == 0) {
            particles_ptr->algorithm = "cutoff";
            std::shared_ptr<Calc> ljCalc = std::make_shared<LJ>();
            particles_ptr->cutoff = check_and_assign_value<double>(doc["LJ"], "cutoff");
            particles_ptr->cutoff_squared = particles_ptr->cutoff * particles_ptr->cutoff;
            calc_manager_ptr->addCalc(ljCalc);
        }
        else if (algorithm.compare("verlet_list") == 0) {
            particles_ptr->algorithm = "verlet_list";
            std::shared_ptr<Calc> ljCalc = std::make_shared<LJ_verlet>();
            particles_ptr->cutoff = check_and_assign_value<double>(doc["LJ"], "cutoff");
            particles_ptr->cutoff_squared = particles_ptr->cutoff * particles_ptr->cutoff;
            calc_manager_ptr->addCalc(ljCalc);
            UseNeighborList = true;
        }
    }

    if (doc["opls"]) {
        auto bonds_ptr = std::make_shared<Bonds>();
        particles_ptr->bonds_ptr = bonds_ptr;
        std::string lammps_data_file = check_and_assign_value<std::string>(doc, "lammps_data_file");
        read_lammps(bonds_ptr, lammps_data_file);
        if(!doc["LJ"])
            Kokkos::abort("ERROR: Cannot use OPLS without defining a Lennard-Jones potential!");
        calc_manager_ptr->addCalc(bonds_ptr);
        if (doc["integrator"]["constrained_bonds"]) {
            std::string constrained_bonds = check_and_assign_value<std::string>(doc["integrator"], "constrained_bonds");
            std::vector<int> constrained_bond_indices;

            // Use a stringstream to parse the comma-separated values
            std::stringstream ss(constrained_bonds);
            std::string token;

            while (std::getline(ss, token, ',')) {
                try {
                    int bond_index = std::stoi(token);  // Convert to integer
                    constrained_bond_indices.push_back(bond_index);
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid number in constrained_bonds: " << token << std::endl;
                } catch (const std::out_of_range& e) {
                    std::cerr << "Number out of range in constrained_bonds: " << token << std::endl;
                }
            }

            bonds_ptr->build_constrained_bond_list(constrained_bond_indices);
        } else {
            bonds_ptr->unconstrained_bonds = bonds_ptr->bonds;
        }
    }

    if (doc["coulomb"]) {
        std::string algorithm = check_and_assign_value<std::string>(doc["coulomb"], "algorithm");
        if (algorithm.compare("ewald") == 0) {
            auto coulomb_ptr = std::make_shared<Coulomb_ewald>(doc,*params_ptr);
            calc_manager_ptr->addCalc(coulomb_ptr);
            UseNeighborList = true;
        }
        else if (algorithm.compare("direct") == 0) {
            auto coulomb_ptr = std::make_shared<Coulomb_direct>();
            calc_manager_ptr->addCalc(coulomb_ptr);
            coulomb_ptr->r_c = check_and_assign_value<double>(doc["coulomb"], "cutoff");
            coulomb_ptr->r_c2 = coulomb_ptr->r_c*coulomb_ptr->r_c;
            UseNeighborList = true;
        }
        particles_ptr->cutoff = check_and_assign_value<double>(doc["coulomb"], "cutoff");
        particles_ptr->cutoff_squared = particles_ptr->cutoff * particles_ptr->cutoff;
    }

    if (UseNeighborList) {
        // If we have any pair potentials utilizing a neighbor list we initialize it here
        // and do a first build
        particles_ptr->neighbor_list_used = true;
        if (doc["opls"]) {
            particles_ptr->neighbor_list = new Neighbor_list_bonds();
        }
        else {
            particles_ptr->neighbor_list = new Neighbor_list();
        }
        // Set frequency of neighbor list build
        if (doc["particles"]["update_every"]) {
            particles_ptr->neighbor_list->update_every = check_and_assign_value<int>(doc["particles"], "update_every");
        }
        else particles_ptr->neighbor_list->update_every = 15; // default rebuild after 15 steps
        particles_ptr->neighbor_list->init_verlet_list(doc,*particles_ptr);
        // set moves_since_last_update so that the build actually gets triggered
        particles_ptr->neighbor_list->moves_since_last_update = particles_ptr->neighbor_list->update_every - 1;
        particles_ptr->neighbor_list->build_verlet_list(*particles_ptr);
    }
}

void Input_reader::read_lammps(std::shared_ptr<Bonds> bonds_ptr, const std::string& filename) {
    // This function reads in lammps-style data files and extracts force-field information (bonds,angles etc.)
    std::ifstream infile(filename);
    std::string line;

    int numBonds = 0, numAngles = 0, numDihedrals = 0;
    int numBondTypes = 0, numAngleTypes = 0, numDihedralTypes = 0;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string keyword;
        int count;
        
        // Read the entire line and check for keywords
        if (iss >> count) {
            std::getline(iss, keyword);  // Get the rest of the line as the keyword

            // Trim leading whitespace from keyword
            keyword = keyword.substr(keyword.find_first_not_of(" \t"));

            if (keyword == "bonds") {
                numBonds = count;
            } else if (keyword == "angles") {
                numAngles = count;
            } else if (keyword == "dihedrals") {
                numDihedrals = count;
            } else if (keyword == "atom types") {
                // Just skipping atom types for now
            } else if (keyword == "bond types") {
                numBondTypes = count;
            } else if (keyword == "angle types") {
                numAngleTypes = count;
            } else if (keyword == "dihedral types") {
                numDihedralTypes = count;
            }
        }
    }

    bonds_ptr->bonds = Kokkos::View<Bond*>("bonds", numBonds);
    bonds_ptr->angles = Kokkos::View<Angle*>("angles", numAngles);
    bonds_ptr->dihedrals = Kokkos::View<Dihedral*>("dihedrals", numDihedrals);
    bonds_ptr->bondTypes = Kokkos::View<BondType*>("bondTypes", numBondTypes);
    bonds_ptr->angleTypes = Kokkos::View<AngleType*>("angleTypes", numAngleTypes);
    bonds_ptr->dihedralTypes = Kokkos::View<DihedralType*>("dihedralTypes", numDihedralTypes);

    bonds_ptr->h_bonds = Kokkos::create_mirror_view(bonds_ptr->bonds);
    bonds_ptr->h_angles = Kokkos::create_mirror_view(bonds_ptr->angles);
    bonds_ptr->h_dihedrals = Kokkos::create_mirror_view(bonds_ptr->dihedrals);
    bonds_ptr->h_bondTypes = Kokkos::create_mirror_view(bonds_ptr->bondTypes);
    bonds_ptr->h_angleTypes = Kokkos::create_mirror_view(bonds_ptr->angleTypes);
    bonds_ptr->h_dihedralTypes = Kokkos::create_mirror_view(bonds_ptr->dihedralTypes);

    bool inBondSection = false, inAngleSection = false, inDihedralSection = false;
    bool inBondTypeSection = false, inAngleTypeSection = false, inDihedralTypeSection = false;
    bool inVelocitySection = false;

    int bondIndex = 0, angleIndex = 0, dihedralIndex = 0;
    int bondTypeIndex = 0, angleTypeIndex = 0, dihedralTypeIndex = 0;

    infile.clear();  // Reset the stream to start reading again
    infile.seekg(0); // Go back to the beginning of the file

    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);

        // Section identification
        if (line.find("Bond Coeffs") != std::string::npos) {
            inBondTypeSection = true;
            inAngleTypeSection = false;
            inDihedralTypeSection = false;
            inDihedralSection = false;
            inBondSection = false;
            inAngleSection = false;
            inVelocitySection = false;
            continue;
        }
        if (line.find("Angle Coeffs") != std::string::npos) {
            inAngleTypeSection = true;
            inBondTypeSection = false;
            inDihedralTypeSection = false;
            inDihedralSection = false;
            inBondSection = false;
            inAngleSection = false;
            inVelocitySection = false;
            continue;
        }
        if (line.find("Dihedral Coeffs") != std::string::npos) {
            inAngleTypeSection = false;
            inBondTypeSection = false;
            inDihedralTypeSection = true;
            inDihedralSection = false;
            inBondSection = false;
            inAngleSection = false;
            inVelocitySection = false;
            continue;
        }
        if (line.find("Bonds") != std::string::npos) {
            inBondSection = true;
            inAngleSection = false;
            inDihedralTypeSection = false;
            inDihedralSection = false;
            inBondTypeSection = false;
            inAngleTypeSection = false;
            inVelocitySection = false;
            continue;
        }
        if (line.find("Angles") != std::string::npos) {
            inAngleSection = true;
            inBondSection = false;
            inDihedralTypeSection = false;
            inDihedralSection = false;
            inBondTypeSection = false;
            inAngleTypeSection = false;
            inVelocitySection = false;
            continue;
        }
        if (line.find("Dihedrals") != std::string::npos) {
            inBondSection = false;
            inAngleSection = false;
            inDihedralTypeSection = false;
            inDihedralSection = true;
            inBondTypeSection = false;
            inAngleTypeSection = false;
            inVelocitySection = false;
            continue;
        }
        if (line.find("Velocities") != std::string::npos) {
            inBondSection = false;
            inAngleSection = false;
            inDihedralTypeSection = false;
            inDihedralSection = false;
            inBondTypeSection = false;
            inAngleTypeSection = false;
            inVelocitySection = true;
            // If we read in velocities we do not want to reinitialize them later
            params_ptr->hb_momenta = false;
            continue;
        }


        // Parse bond type data
        if (inBondTypeSection && bondTypeIndex < numBondTypes) {
            int type;
            double k, r0;
            if (iss >> type >> k >> r0) {
                bonds_ptr->h_bondTypes(bondTypeIndex).type = type - 1; // lammps indices start at 1
                bonds_ptr->h_bondTypes(bondTypeIndex).k = k*kcaltointernal; // convert from kcal/mol to internal units
                bonds_ptr->h_bondTypes(bondTypeIndex).r0 = r0;
                bondTypeIndex++;
            }
        }

        // Parse angle type data
        if (inAngleTypeSection && angleTypeIndex < numAngleTypes) {
            int type;
            double k, theta0;
            if (iss >> type >> k >> theta0) {
                bonds_ptr->h_angleTypes(angleTypeIndex).type = type - 1; // lammps indices start at 1
                bonds_ptr->h_angleTypes(angleTypeIndex).k = k*kcaltointernal;// convert from kcal/mol to internal units
                bonds_ptr->h_angleTypes(angleTypeIndex).theta0 = theta0 * M_PI/180.0;
                angleTypeIndex++;
            }
        }

        // Parse dihedral type data
        if (inDihedralTypeSection && dihedralTypeIndex < numDihedralTypes) {
            int type;
            double k1, k2, k3, k4;
            if (iss >> type >> k1 >> k2 >> k3 >> k4) {
                bonds_ptr->h_dihedralTypes(dihedralTypeIndex).type = type - 1; // lammps indices start at 1
                // for some reason lammps files include the usual factor of 0.5 into all k-values
                // except for the dihedrals so we have to explicitly add it here
                bonds_ptr->h_dihedralTypes(dihedralTypeIndex).k1 = 0.5*k1*kcaltointernal;// convert from kcal/mol to internal units
                bonds_ptr->h_dihedralTypes(dihedralTypeIndex).k2 = 0.5*k2*kcaltointernal;// convert from kcal/mol to internal units
                bonds_ptr->h_dihedralTypes(dihedralTypeIndex).k3 = 0.5*k3*kcaltointernal;// convert from kcal/mol to internal units
                bonds_ptr->h_dihedralTypes(dihedralTypeIndex).k4 = 0.5*k4*kcaltointernal;// convert from kcal/mol to internal units
                dihedralTypeIndex++;
            }
        }

        // Parse bond data
        if (inBondSection && bondIndex < numBonds) {
            int id, type, atom1, atom2;
            if (iss >> id >> type >> atom1 >> atom2) {
                bonds_ptr->h_bonds(bondIndex).id = id;
                bonds_ptr->h_bonds(bondIndex).type = type - 1;
                bonds_ptr->h_bonds(bondIndex).atom1 = atom1 - 1;
                bonds_ptr->h_bonds(bondIndex).atom2 = atom2 - 1;
                bondIndex++;
            }
        }

        // Parse angle data
        if (inAngleSection && angleIndex < numAngles) {
            int id, type, atom1, atom2, atom3;
            if (iss >> id >> type >> atom1 >> atom2 >> atom3) {
                bonds_ptr->h_angles(angleIndex).id = id;
                bonds_ptr->h_angles(angleIndex).type = type - 1; // lammps indices start at 1
                bonds_ptr->h_angles(angleIndex).atom1 = atom1 - 1; // lammps indices start at 1
                bonds_ptr->h_angles(angleIndex).atom2 = atom2 - 1; // lammps indices start at 1
                bonds_ptr->h_angles(angleIndex).atom3 = atom3 - 1; // lammps indices start at 1
                angleIndex++;
            }
        }

        // Parse dihedral data
        if (inDihedralSection && dihedralIndex < numDihedrals) {
            int id, type, atom1, atom2, atom3, atom4;
            if (iss >> id >> type >> atom1 >> atom2 >> atom3 >> atom4) {
                bonds_ptr->h_dihedrals(dihedralIndex).id = id;
                bonds_ptr->h_dihedrals(dihedralIndex).type = type - 1;
                bonds_ptr->h_dihedrals(dihedralIndex).atom1 = atom1 - 1;
                bonds_ptr->h_dihedrals(dihedralIndex).atom2 = atom2 - 1;
                bonds_ptr->h_dihedrals(dihedralIndex).atom3 = atom3 - 1;
                bonds_ptr->h_dihedrals(dihedralIndex).atom4 = atom4 - 1;
                dihedralIndex++;
            }
        }

        if (inVelocitySection) {
            double vx,vy,vz;
            int id;
            if (iss >> id >> vx >> vy >> vz) {
                id -= 1;
                double m = particles_ptr->atom_type_list[particles_ptr->id[id]].mass;
                particles_ptr->h_p(id,0) = m*vx;
                particles_ptr->h_p(id,1) = m*vy;
                particles_ptr->h_p(id,2) = m*vz;
            }
        }
    }

    // Copy data to device
    Kokkos::deep_copy(bonds_ptr->bonds, bonds_ptr->h_bonds);
    Kokkos::deep_copy(bonds_ptr->angles, bonds_ptr->h_angles);
    Kokkos::deep_copy(bonds_ptr->dihedrals, bonds_ptr->h_dihedrals);
    Kokkos::deep_copy(bonds_ptr->bondTypes, bonds_ptr->h_bondTypes);
    Kokkos::deep_copy(bonds_ptr->angleTypes, bonds_ptr->h_angleTypes);
    Kokkos::deep_copy(bonds_ptr->dihedralTypes, bonds_ptr->h_dihedralTypes);
    Kokkos::deep_copy(particles_ptr->p, particles_ptr->h_p);

    /*std::cout << "Number of Bonds: " << numBonds << std::endl;
    std::cout << "Number of Angles: " << numAngles << std::endl;
    std::cout << "Number of Dihedrals: " << numDihedrals << std::endl;
    std::cout << "Actual Bonds Read: " << bondIndex << std::endl;
    std::cout << "Actual Angles Read: " << angleIndex << std::endl;
    std::cout << "Actual Dihedrals Read: " << dihedralIndex << std::endl;
    std::cout << "Number of Bond Types: " << numBondTypes << std::endl;
    std::cout << "Number of Angle Types: " << numAngleTypes << std::endl;
    std::cout << "Number of Dihedral Types: " << numDihedralTypes << std::endl;*/

    /*for (int i = 0; i < particles_ptr->h_x.extent(0); i++) {
        printf("atom %d: %f %f %f \n", i, particles_ptr->h_x(i,0),particles_ptr->h_x(i,1),particles_ptr->h_x(i,2));
    }*/

    /*for (int i = 0; i < bonds_ptr->bonds.extent(0); i++) {
        printf("bond number: %d \n", i);
        printf("bond type: %d \n", bonds_ptr->h_bonds(i).type+1);
        printf("atom1: %d atom2: %d k: %f r0: %f\n", bonds_ptr->h_bonds(i).atom1+1, bonds_ptr->h_bonds(i).atom2+1, bonds_ptr->h_bondTypes(bonds_ptr->h_bonds(i).type).k, bonds_ptr->h_bondTypes(bonds_ptr->h_bonds(i).type).r0);
    }*/
    
    /*for (int i = 0; i < bonds_ptr->bondTypes.extent(0); i++) {
        printf("bond type number: %d \n", i);
        printf("type: %d k: %f r0: %f\n", bonds_ptr->h_bondTypes(i).type, bonds_ptr->h_bondTypes(i).k, bonds_ptr->h_bondTypes(i).r0);
    }*/
    
    /*for (int i = 0; i < bonds_ptr->angles.extent(0); i++) {
        printf("angle number: %d \n", i);
        printf("atom1: %d atom2: %d atom3: %d k: %f theta0: %f\n", bonds_ptr->h_angles(i).atom1, bonds_ptr->h_angles(i).atom2, bonds_ptr->h_angles(i).atom3, bonds_ptr->h_angleTypes(bonds_ptr->h_angles(i).type-1).k, bonds_ptr->h_angleTypes(bonds_ptr->h_angles(i).type-1).theta0);
    }*/
    /*printf("EXTENT: %d", bonds_ptr->dihedrals.extent(0));
    for (int i = 0; i < bonds_ptr->dihedrals.extent(0); i++) {
        printf("dihedral number: %d \n", i);
        printf("atom1: %d atom2: %d atom3: %d atom4: %d\n", bonds_ptr->h_dihedrals(i).atom1, bonds_ptr->h_dihedrals(i).atom2, bonds_ptr->h_dihedrals(i).atom3, bonds_ptr->h_dihedrals(i).atom4);
    }*/

    /*for (int i = 0; i < bonds_ptr->dihedralTypes.extent(0); i++) {
        printf("dihedral number: %d \n", i);
        printf("atom1: %f atom2: %f atom3: %f atom4: %f\n", bonds_ptr->h_dihedralTypes(i).k1, bonds_ptr->h_dihedralTypes(i).k2, bonds_ptr->h_dihedralTypes(i).k3, bonds_ptr->h_dihedralTypes(i).k4);
    }*/
}