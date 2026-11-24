#include "general.hpp"
#include <stdexcept>
#include <string>

// Read cycles, a and b; n_a and n_b are the expected lengths of a and b.
// AB and BA have the lengths swapped, so this catches coefficients of the other ordering.
static void read_coefficients(YAML::Node integrator, const std::string& name, int& cycles,
                              std::vector<double>& a, std::vector<double>& b, bool drift_first) {
    cycles = integrator["cycles"].as<int>();
    a = integrator["a"].as<std::vector<double>>();
    b = integrator["b"].as<std::vector<double>>();
    const size_t n_a = drift_first ? cycles + 1 : cycles;
    const size_t n_b = drift_first ? cycles : cycles + 1;
    if (a.size() != n_a || b.size() != n_b) {
        throw std::runtime_error("Error: " + name + " with cycles = " + std::to_string(cycles) + " needs " +
                                 std::to_string(n_a) + " drift coefficients a and " +
                                 std::to_string(n_b) + " kick coefficients b, got " +
                                 std::to_string(a.size()) + " and " + std::to_string(b.size()));
    }
}

// ---------------- AB: drift first ----------------

GENERAL_AB::GENERAL_AB(YAML::Node doc, params_class params)
    : integrator_type(doc, params) {
    read_coefficients(doc["integrator"], "GENERAL_AB", cycles, a, b, true);
}

void GENERAL_AB::integrate() {
    // First position step
    particles->update_positions(dt * a[0]);
    if (particles->neighbor_list_used) {
        particles->neighbor_list->build_verlet_list(*particles);
    }
    for (int i = 0; i < steps; i++) {
        for (int j = 0; j < cycles; j++) {
            // Momenta step
            calc_manager->compute_force();
            Kokkos::fence();
            particles->update_momenta(dt * b[j]);
            // Position step; the last one of a step is merged with the first one of the next step
            double c = a[j + 1];
            if (j == cycles - 1 && i < steps - 1) c = 2*a[0];
            particles->update_positions(dt * c);
            if (particles->neighbor_list_used) {
                particles->neighbor_list->build_verlet_list(*particles);
            }
        }
    }
}

// ---------------- BA: kick first ----------------

GENERAL_BA::GENERAL_BA(YAML::Node doc, params_class params)
    : integrator_type(doc, params) {
    read_coefficients(doc["integrator"], "GENERAL_BA", cycles, a, b, false);
}

void GENERAL_BA::integrate() {
    // First part of the chain
    for (int j = 0; j < cycles; j++) {
        // Momenta step
        calc_manager->compute_force();
        Kokkos::fence();
        particles->update_momenta(dt * b[j]);
        // Position step
        particles->update_positions(dt * a[j]);
        if (particles->neighbor_list_used) {
            particles->neighbor_list->build_verlet_list(*particles);
        }
    }
    // Rest of the chain (n-1 steps); the first kick is merged with the last kick of the previous step
    for (int i = 0; i < steps - 1; i++) {
        // Momenta step
        calc_manager->compute_force();
        Kokkos::fence();
        particles->update_momenta(dt * 2*b[0]);
        // Position step
        particles->update_positions(dt * a[0]);
        if (particles->neighbor_list_used) {
            particles->neighbor_list->build_verlet_list(*particles);
        }
        for (int j = 1; j < cycles; j++) {
            // Momenta step
            calc_manager->compute_force();
            Kokkos::fence();
            particles->update_momenta(dt * b[j]);
            // Position step
            particles->update_positions(dt * a[j]);
            if (particles->neighbor_list_used) {
                particles->neighbor_list->build_verlet_list(*particles);
            }
        }
    }
    // Final momenta step
    calc_manager->compute_force();
    Kokkos::fence();
    particles->update_momenta(dt * b.back());
}
