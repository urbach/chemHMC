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
#include "read_infile.hpp"


particles_type::particles_type(YAML::Node doc, params_class params_to_copy) {

    N = check_and_assign_value<int>(doc, "N");
    mass = check_and_assign_value<double>(doc, "mass");
    beta = check_and_assign_value<double>(doc, "beta");

    std::cout << "constructor particles_type" << std::endl;
    params = params_to_copy;
    rand_pool.init(params.seed, N);
    std::cout << "random pool initialised" << std::endl;

    std::cout << "N:" << N << std::endl;
    std::cout << "mass:" << mass << std::endl;
    std::cout << "beta:" << beta << std::endl;

    initHostMirror = false;
}


// contructor
identical_particles::identical_particles(YAML::Node doc, params_class params_to_copy) : particles_type(doc, params_to_copy) {

    std::cout << "partilces_type:" << std::endl;
    std::cout << "name:" << name << std::endl;

}




void particles_type::InitX() {
    x = Kokkos::View<double* [dim_space]>("x", N);
    p = Kokkos::View<double* [dim_space]>("p", N);
    if (params.StartCondition == "cold") {
        // we define a cold setput when 
        Kokkos::parallel_for("cold initialization", N, KOKKOS_LAMBDA(int i){

            double N3 = pow(N, 1. / 3.);
            int iz = (int)i / (N3 * N3);
            int iy = (int)(i - iz * N3 * N3) / (N3);
            int ix = (int)(i - iz * N3 * N3 - iy * N3);

            x(i, 0) = params.L[0] * (ix / N3);
            x(i, 1) = params.L[1] * (iy / N3);
            x(i, 2) = params.L[2] * (iz / N3);
        });
    }
    if (params.StartCondition == "hot") {
        Kokkos::parallel_for("hot initialization", N, KOKKOS_LAMBDA(int i){
            gen_type rgen = rand_pool.get_state(i);
            x(i, 1) = rgen.drand() * params.L[0];
            x(i, 2) = rgen.drand() * params.L[1];
            x(i, 3) = rgen.drand() * params.L[2];
            rand_pool.free_state(rgen);
        });
    }
};

void particles_type::printx() {
    if (!initHostMirror) {
        h_x = Kokkos::create_mirror_view(x);
        initHostMirror = true;
    }
    Kokkos::deep_copy(h_x, x);
    for (int i = 0; i < N;i++)
        printf("particle(%d)=%-20.12g %-20.12g %-20.12g\n", i, h_x(i, 0), h_x(i, 1), h_x(i, 2));
}
