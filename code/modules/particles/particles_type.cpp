#include "particles_type.hpp"

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


particles_type::particles_type(YAML::Node doc, params_class params)  {

    std::cout << "constructor particles_type" << std::endl;
    L[0] = params.L[0];
    L[1] = params.L[1];
    L[2] = params.L[2];

    N = params.N;
    rand_pool.init(params.seed, N);
    std::cout << "random pool initialised" << std::endl;

}