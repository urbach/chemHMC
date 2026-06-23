#include "OMF4.hpp"

OMF4::OMF4(YAML::Node doc, params_class params) :
    integrator_type(doc, params),
    rho(0.2539785108410595),
    theta(-0.03230286765269967),
    vartheta(0.08398315262876693),
    lambda(0.6822365335719091),
    dtau(dt),
    eps{ rho * dtau, lambda * dtau,
                 theta * dtau, 0.5 * (1 - 2. * (lambda + vartheta)) * dtau,
                 (1 - 2. * (theta + rho)) * dtau, 0.5 * (1 - 2. * (lambda + vartheta)) * dtau,
                 theta * dtau, lambda * dtau,
                 rho * dtau, 2 * vartheta * dtau } {
}

void OMF4::integrate() {

    // initial half-step for the momenta
    calc_manager->compute_force();
    Kokkos::fence();
    particles->update_momenta(0.5 * eps[9]);

    // nsteps-1 full steps
    for (size_t i = 1; i < steps - 1; i++) {
        for (size_t j = 0; j < 5; j++) {
            particles->update_positions(eps[2 * j]);
            if (particles->neighbor_list_used) {
                particles->neighbor_list->build_verlet_list(*particles);
            }
            calc_manager->compute_force();
            Kokkos::fence();
            particles->update_momenta(eps[2 * j + 1]);
        }
    }
    // almost one more full step
    for (size_t j = 0; j < 4; j++) {
        particles->update_positions(eps[2 * j]);
        if (particles->neighbor_list_used) {
            particles->neighbor_list->build_verlet_list(*particles);
        }
        calc_manager->compute_force();
        Kokkos::fence();
        particles->update_momenta(eps[2 * j + 1]);
    }
    particles->update_positions(eps[8]);
    if (particles->neighbor_list_used) {
        particles->neighbor_list->build_verlet_list(*particles);
    }
    // final half-step in the momenta
    calc_manager->compute_force();
    Kokkos::fence();
    particles->update_momenta(0.5 * eps[9]);
}