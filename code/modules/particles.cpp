#include "particles.hpp"

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
#include "read_infile.hpp"


particles_type::particles_type(YAML::Node doc) : params(doc) {


    N = check_and_assign_value<int>(doc["particles"], "N");

    std::cout << "constructor particles_type" << std::endl;

    rand_pool.init(params.seed, N);
    L[0] = params.L[0];
    L[1] = params.L[1];
    L[2] = params.L[2];
    std::cout << "random pool initialised" << std::endl;

    std::cout << "N:" << N << std::endl;

    fileout = NULL;
    nameout = check_and_assign_value<std::string>(doc, "output_file");
    fileout = fopen(nameout.c_str(), "ab");
    if (fileout == NULL || nameout.length() <= 0 || nameout.compare("null") == 0) {
        printf("unable to open file %s\n", nameout.c_str());
        Kokkos::abort("abort");
    }

    initHostMirror = false;
}
