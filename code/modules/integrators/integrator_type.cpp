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