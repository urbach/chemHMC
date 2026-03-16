#include "general.hpp"

GENERAL::GENERAL(YAML::Node doc, params_class params)
    : integrator_type(doc, params) {
    YAML::Node integrator = doc["integrator"];
    cycles = integrator["cycles"].as<int>();
    a = integrator["a"].as<std::vector<double>>();
    b = integrator["b"].as<std::vector<double>>();
}

void GENERAL::integrate() {
    // First part of the chain
    for (int j = 0; j < cycles; j++) {
        // Momenta step
        calc_manager->compute_force();
        Kokkos::fence();
        particles->update_momenta(dt * a[j]);
        // Position step
        particles->update_positions(dt * b[j]);
        if (particles->neighbor_list_used) {
            particles->neighbor_list->build_verlet_list(*particles);
        }
    }
    
    // Rest of the chain (n-1 steps)
    for (size_t i = 0; i < steps - 1; i++) {
        // Momenta step
        calc_manager->compute_force();
        Kokkos::fence();
        particles->update_momenta(dt * 2*a[0]);
        // Position step
        particles->update_positions(dt * b[0]);
        if (particles->neighbor_list_used) {
            particles->neighbor_list->build_verlet_list(*particles);
        }
        for (int j = 1; j < cycles; j++) {
            // Momenta step
            calc_manager->compute_force();
            Kokkos::fence();
            particles->update_momenta(dt * a[j]);
            // Position step
            particles->update_positions(dt * b[j]);
            if (particles->neighbor_list_used) {
                particles->neighbor_list->build_verlet_list(*particles);
            }
        }
    }
    // Final momenta step
    calc_manager->compute_force();
    Kokkos::fence();
    particles->update_momenta(dt * a[-1]);
}