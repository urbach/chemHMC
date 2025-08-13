#include "OMF2.hpp"

OMF2::OMF2(YAML::Node doc, params_class params) :
    integrator_type(doc, params), lambda(0.1938), oneminus2lambda(1. - 2. * lambda) {
    // todo: check lambda= 0.1931833275037836
}


void OMF2::integrate() {

    // initial half-step for the  momenta
    calc_manager->compute_force();
    Kokkos::fence();
    particles->update_momenta(lambda * dt);

    // nsteps-1 full steps
    for (size_t i = 0; i < steps - 1; i++) {
        particles->update_positions(dt / 2.);
        calc_manager->compute_force();
        Kokkos::fence();
        particles->update_momenta(oneminus2lambda * dt);
        particles->update_positions(dt / 2.);
        particles->neighbor_list->build_verlet_list(*particles);
        calc_manager->compute_force();
        Kokkos::fence();
        particles->update_momenta(2. * lambda * dt);
    }
    // final step
    particles->update_positions(dt / 2.);
    calc_manager->compute_force();
    Kokkos::fence();
    particles->update_momenta(oneminus2lambda * dt);
    particles->update_positions(dt / 2.);
    calc_manager->compute_force();
    Kokkos::fence();
    particles->update_momenta(lambda * dt);
}