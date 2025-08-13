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