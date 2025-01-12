#include "Input_reader.hpp"
#include "read_infile.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

// Constructor: Accepts a pointer to Parameters
Input_reader::Input_reader(params_class* params, integrator_type*& integrator) 
                        : params_ptr(params), integrator_ptr(integrator) {
    if (!params) {
        throw std::runtime_error("Error: Null pointer passed as params to Input_reader constructor.");
    }
}

// Validates the existence of the file
void Input_reader::validate_file(const std::string& file_path) const {
    std::ifstream file(file_path);
    if (!file.good()) {
        throw std::runtime_error("Error: Unable to open file: " + file_path);
    }
}

// Template for safely extracting and converting YAML values
template <typename T>
T Input_reader::check_and_assign_value(const YAML::Node& node, const char* tag) const {
    if (!node[tag]) {
        throw std::runtime_error(std::string("Error: Missing tag in YAML: ") + tag);
    }
    try {
        return node[tag].as<T>();
    } catch (const YAML::TypedBadConversion<T>& e) {
        throw std::runtime_error(std::string("Error: Type mismatch for tag: ") + tag);
    }
}

// Parse the input file and populate the Parameters struct
void Input_reader::parse_input(int argc, char** argv) {
    int opt = -1;
    YAML::Node doc;
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
    params_ptr->start_configuration_file = check_and_assign_value<std::string>(doc, "start_configuration_file");
    params_ptr->nameout = check_and_assign_value<std::string>(doc, "output_file");
    params_ptr->parameter_file = check_and_assign_value<std::string>(doc, "parameter_file");
    params_ptr->Ntrajectories = check_and_assign_value<int>(doc, "Ntrajectories");
    params_ptr->thermalization_steps = check_and_assign_value<int>(doc, "thermalization_steps");
    params_ptr->save_every = check_and_assign_value<int>(doc, "save_every");
    params_ptr->print_info_every = check_and_assign_value<int>(doc, "print_info_every");
    params_ptr->seed = check_and_assign_value<int>(doc, "seed");
    
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
        } else {
            throw std::runtime_error("Error: Invalid integrator name: " + name);
        }
    } else {
        throw std::runtime_error("Error: No integrator specified in input file.");
    }
    integrator_ptr->dt = check_and_assign_value<double>(doc["integrator"], "dt");
    integrator_ptr->steps = check_and_assign_value<int>(doc["integrator"], "steps");
}