#include <iostream> 
#include "integrator.hpp"
#include "Parameters.hpp"
#include "Input_reader.hpp"
#include "particles.hpp"

integrator_type::integrator_type(YAML::Node doc, params_class params) {
}

void integrator_type::set_calc_manager(Calc_Manager& calc_manager_ref) {
    // The integrator needs to save a reference to the calc_manager to be able
    // to trigger force calculations and neighbour-list builds
    calc_manager = &calc_manager_ref;
}

VELOCITY_VERLET::VELOCITY_VERLET(YAML::Node doc, params_class params) : integrator_type(doc, params) {}

void VELOCITY_VERLET::integrate() {
    calc_manager->compute_force();
    Kokkos::fence();
    for (size_t i = 0; i < steps; i++) {
        particles->update_momenta(dt / 2.);
        particles->update_positions(dt);
        calc_manager->compute_force();
        Kokkos::fence();
        particles->update_momenta(dt / 2.);
    }
}

LEAP::LEAP(YAML::Node doc, params_class params) : integrator_type(doc, params) {}

// binomial distribution with average n*p= average_steps
void integrator_type::set_binomial_steps(std::mt19937_64 &gen64) {
    int K = 2;
    // nouber of steps
    int n = K * average_steps;
    // success propability 
    double p = 1. / ((double)(K));
    steps = 0;
    for (int i = 0;i < n;i++) {
        double r = ((double)gen64() - gen64.min()) / (gen64.max() - gen64.min());// random number from 0 to 1
        if (r < p)
            steps++;
    }
}

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

//////////////////////////////////////////////////////////////////////////////
// OMF2
//////////////////////////////////////////////////////////////////////////////
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
//////////////////////////////////////////////////////////////////////////////
// OMF4 integration scheme
//////////////////////////////////////////////////////////////////////////////
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
    particles->update_momenta(0.5 * eps[9]);

    // nsteps-1 full steps
    for (size_t i = 1; i < steps - 1; i++) {
        for (size_t j = 0; j < 5; j++) {
            particles->update_positions(eps[2 * j]);
            particles->update_momenta(eps[2 * j + 1]);
        }
    }
    // almost one more full step
    for (size_t j = 0; j < 4; j++) {
        particles->update_positions(eps[2 * j]);
        particles->update_momenta(eps[2 * j + 1]);
    }
    particles->update_positions(eps[8]);
    // final half-step in the momenta
    particles->update_momenta(0.5 * eps[9]);
}
