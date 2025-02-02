#include "leapfrog.hpp"

void LEAP::integrate() {
    // initial half-step for the  momenta
    calc_manager->compute_force();
    Kokkos::fence();
    particles->update_momenta(dt / 2.);
    //
    // first full step for the position
    particles->update_positions(dt);
    // nsteps-1 full steps
    for (size_t i = 0; i < steps - 1; i++) {
        calc_manager->compute_force();
        Kokkos::fence();
        particles->update_momenta(dt);
        particles->update_positions(dt);
    }
    // final half-step for the momenta
    calc_manager->compute_force();
    Kokkos::fence();
    particles->update_momenta(dt / 2.);
}