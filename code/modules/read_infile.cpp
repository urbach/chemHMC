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
#include "particles.hpp"
#include "fmt/core.h"



template<class T>
T check_and_assign_value(YAML::Node doc, const char* tag) {
    if (!doc[tag]) {
        fmt::print("error: tag '{}' is not present in the input file\n", tag);
        std::cout << "infile structure:" << std::endl;
        std::cout << doc << std::endl;
        Kokkos::abort("params not found");
    }

    try {
        return doc[tag].as<T>();
    } catch (YAML::TypedBadConversion<T>) {
        Kokkos::abort(fmt::format ("Incorrect input type for '{}'.", tag).c_str());
    }
}
template double check_and_assign_value<double>(YAML::Node, const char*);
template int check_and_assign_value<int>(YAML::Node, const char*);
template std::string check_and_assign_value<std::string>(YAML::Node, const char*);



params_class::params_class(YAML::Node doc) {

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