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
    L[0] = params.L[0];
    L[1] = params.L[1];
    L[2] = params.L[2];


    std::cout << "N:" << N << std::endl;
    initHostMirror = false;
    padding = N < 1000 ? 64 : 1;
    hs = typename state_data_type::HostMirror("hostMirror_rng_state", N, padding);
    rand_pool.init(params.seed, N);
    if (params.append)     load_device_rng();
    std::cout << "random pool initialised" << std::endl;

}

void particles_type::save_device_rng() {
    rand_pool.return_rng_state(hs);
    FILE* f;
    f = fopen(params.rng_device_state.c_str(), "w+");
    printf("Saving rnd device...  %d %d\n",N,padding);
    int i=fwrite(&hs(0, 0), sizeof(uint64_t), N * padding, f);
    fclose(f);
}
void particles_type::load_device_rng() {
    FILE* f;
    f = fopen(params.rng_device_state.c_str(), "r");
    int i=fread(&hs(0, 0), sizeof(uint64_t), N * padding, f);
    if (i!=N*padding) Kokkos::abort("invalid rng_device_state file\n");
    rand_pool.load_rng_state(hs);
    fclose(f);
}