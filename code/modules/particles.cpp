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


// contructor
identical_particles::identical_particles(YAML::Node doc) {


    N = check_and_assign_value<int>(doc, "N");
    mass = check_and_assign_value<double>(doc, "mass");
    beta = check_and_assign_value<double>(doc, "beta");

    std::cout << "partilces_type:" << std::endl;
    std::cout << "name:" << name << std::endl;
    std::cout << "N:" << N << std::endl;
    std::cout << "mass:" << mass << std::endl;
    std::cout << "beta:" << beta << std::endl;

}




void particles_type::InitX(params_class params) {
    x = Kokkos::View<double* [dim_space]>("x", N);
    p = Kokkos::View<double* [dim_space]>("p", N);
    if (params.StartCondition == "cold") {
        Kokkos::parallel_for("cold initialization", N, KOKKOS_LAMBDA(int i){
            x(i, 0) = params.L[0] * (((double)i) / N);
            x(i, 1) = params.L[1] * (((double)i) / N);
            x(i, 2) = params.L[2] * (((double)i) / N);
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

KOKKOS_INLINE_FUNCTION
void particles_type::operator() (cold, const int& i) const {
    // get a random generatro from the pool



};


// void identical_particles::hb() {
//     Kokkos::parallel_for(Kokkos::RangePolicy<hbTag>(0, N), *this);
// }

// KOKKOS_FUNCTION
// void identical_particles::operator() (hbTag, const int i) const {
//     // get a random generatro from the pool
//     gen_type rgen = rand_pool.get_state(i);
//     p(i, 0) =rgen.drand();
//     p(i, 0) =rgen.drand();
//     p(i, 0) =rgen.drand();

//     rand_pool.free_state(rgen);
// }