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