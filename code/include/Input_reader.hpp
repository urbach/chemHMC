#ifndef INPUT_READER_H
#define INPUT_READER_H

#include "Parameters.hpp"
#include "integrator.hpp"
#include "read_infile.hpp"
#include "yaml-cpp/yaml.h"
#include <string>

class Input_reader {
public:
    // Constructor accepts a pointer to Parameters struct
    explicit Input_reader(params_class* params, integrator_type*& integrator);

    // Method to parse and populate data from a file
    void parse_input(int argc, char** argv);

private:
    params_class* params_ptr;
    integrator_type*& integrator_ptr;

    void parse_simulation_parameters(YAML::Node& doc);
    void parse_integrator_options(YAML::Node& doc);

    // Helper method to validate file existence
    void validate_file(const std::string& file_path) const;

    // Template method for safely extracting values from YAML
    template <typename T>
    T check_and_assign_value(const YAML::Node& node, const char* tag) const;
};

#endif