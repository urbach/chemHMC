#include "read_infile.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "global.hpp"
#include "yaml-cpp/yaml.h"
#include "particles_type.hpp"



template<class T>
T check_and_assign_value(YAML::Node doc, const char* tag) {
    if (!doc[tag]) {
        printf("error: tag %s is not present in the input file\n", tag);
        std::cout << "infile structure:" << std::endl;
        std::cout << doc << std::endl;
        Kokkos::abort("params not found");
    }

    try {
        return doc[tag].as<T>();
    }
    catch (YAML::TypedBadConversion<T>) {
        printf("error: impossible to read tag %s\n", tag);
        Kokkos::abort("Incorrect input type");
    }
}
// int specialization
template<>
int check_and_assign_value(YAML::Node doc, const char* tag) {
    if (!doc[tag]) {
        printf("error: tag %s is not present in the input file\n", tag);
        std::cout << "infile structure:" << std::endl;
        std::cout << doc << std::endl;
        Kokkos::abort("params not found");
    }

    try {
        return ((int)stod(doc[tag].as<std::string>()));
    }
    catch (YAML::TypedBadConversion<std::string>) {
        printf("error: impossible to read tag %s\n", tag);
        Kokkos::abort("Incorrect input type");
    }
}
template double check_and_assign_value<double>(YAML::Node, const char*);
// template int check_and_assign_value<int>(YAML::Node, const char*);
template std::string check_and_assign_value<std::string>(YAML::Node, const char*);

inline bool file_exist(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

void error_if_file_exist(const std::string& name) {
    if (file_exist(name)) {
        printf("error: output file %s exist but in the input file\n append=false found\n", name.c_str());
        Kokkos::abort("aborting");
    }
}
void error_if_can_not_open_file_to_read(const std::string& name) {
    FILE* f = NULL;
    f = fopen(name.c_str(), "r");
    if (f == NULL || name.length() <= 0 || name.compare("null") == 0) {
        printf("unable to open file %s\n", name.c_str());
        Kokkos::abort("abort");
    }
    fclose(f);
}
void error_if_can_not_open_file_to_write(const std::string& name) {
    FILE* f = NULL;
    f = fopen(name.c_str(), "w+");
    if (f == NULL || name.length() <= 0 || name.compare("null") == 0) {
        printf("unable to open file %s\n", name.c_str());
        Kokkos::abort("abort");
    }
    fclose(f);
}

params_class::params_class(YAML::Node doc, bool check_overwrite) {

    // get box dimensions
    L[0] = check_and_assign_value<double>(doc["geometry"], "Lx");
    L[1] = check_and_assign_value<double>(doc["geometry"], "Ly");
    L[2] = check_and_assign_value<double>(doc["geometry"], "Lz");
    seed = check_and_assign_value<int>(doc, "seed");
    StartCondition = check_and_assign_value<std::string>(doc, "StartCondition");

    std::cout << "geometry: " << L[0] << std::endl;
    std::cout << "   Lx: " << L[0] << std::endl;
    std::cout << "   Ly: " << L[1] << std::endl;
    std::cout << "   Lz: " << L[2] << std::endl;
    std::cout << "seed: " << seed << std::endl;
    std::cout << "StartCondition: " << StartCondition << std::endl;
    if (StartCondition == "read") {
        start_configuration_file = check_and_assign_value<std::string>(doc, "start_configuration_file");
    }
    fileout = NULL;
    nameout = check_and_assign_value<std::string>(doc, "output_file");

    parameter_file = check_and_assign_value<std::string>(doc, "parameter_file");  

    printf("parameter_file:");
    printf("parameter_file: %s\n", parameter_file.c_str());

    append = check_and_assign_value<bool>(doc, "append");
    
    // Sanity check
    if (append == true) {
        if (StartCondition != "read") {
            printf("error: append=true so the start condition must be read, while in the inputfile StartCondition=%s\n", StartCondition.c_str());
            Kokkos::abort("aborting");
        }
        if (start_configuration_file != nameout) {
            printf("error: append=true is only supported if the start_configuration_file is the same of the output_file\n");
            printf("start_configuration_file= %s\n", start_configuration_file.c_str());
            printf("output_file             = %s\n", nameout.c_str());
            Kokkos::abort("aborting");
        }
        error_if_can_not_open_file_to_read(parameter_file);
    }
    else {
        if (check_overwrite) {
            error_if_file_exist(nameout);
        }
    }

    fileout = fopen(nameout.c_str(), "ab");
    if (fileout == NULL || nameout.length() <= 0 || nameout.compare("null") == 0) {
        printf("unable to open file %s\n", nameout.c_str());
        Kokkos::abort("abort");
    }
    istart = 0;

    if (doc["particles"]["RDF"]) {
        name_RDF = check_and_assign_value<std::string>(doc["particles"]["RDF"], "output_file");
    }

}

YAML::Node read_params(int argc, char** argv) {
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
        exit(1); // TODO: call Kokkos::finalize
    }
    std::string infilename = argv[opt];
    std::cout << "Trying input file " << infilename << std::endl;

    doc = YAML::LoadFile(infilename);
    return doc;
}