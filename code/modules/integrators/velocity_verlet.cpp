#include "velocity_verlet.hpp"

VELOCITY_VERLET::VELOCITY_VERLET(YAML::Node doc, params_class params) : integrator_type(doc, params) {}

void VELOCITY_VERLET::integrate() {
    calc_manager->compute_force();
    Kokkos::fence();
    for (size_t i = 0; i < steps; i++) {
        particles->update_momenta(dt / 2.);
        particles->update_positions(dt);
        particles->neighbor_list->build_verlet_list(*particles);
        calc_manager->compute_force();
        Kokkos::fence();
        particles->update_momenta(dt / 2.);
    }
}