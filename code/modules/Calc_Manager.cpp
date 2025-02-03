#include "Calc_Manager.hpp"
#include <iostream>

void Calc_Manager::addCalc(std::shared_ptr<Calc> calc) {
    calc_list.push_back(calc);
}

void Calc_Manager::initialize() {
    for (const auto& calc : calc_list) {
        if (calc) { // Check if the pointer is valid
            calc->init( *particles );
        }
    }
}

void Calc_Manager::compute_force() {
    // Set forces to 0
    Kokkos::deep_copy(particles->f, 0.0);
    // Loop over different potentials in calc_list
    for (const auto& calc : calc_list) {
            calc->force( *particles, particles->f );
    }
}

double Calc_Manager::compute_potential() {
    double result = 0;
    // Loop over different potentials in calc_list
    for (const auto& calc : calc_list) {
            result += calc->potential( *particles );
    }
    return result;
}

void Calc_Manager::set_particles(std::shared_ptr<particles_instance> particles_in) {
    particles = particles_in;
}

void Calc_Manager::print_timings() {
    for (const auto& calc : calc_list) {
        calc->print_timings();
    }
}