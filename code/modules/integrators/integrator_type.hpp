#ifndef INTEGRATOR_TYPE_H
#define INTEGRATOR_TYPE_H


#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>
#include "global.hpp"
#include "Parameters.hpp"
#include "particles.hpp"
#include <random>
#include "atom.hpp"
#include "Calc_Manager.hpp"

class integrator_type {

public:
    particles_instance* particles;
    Calc_Manager* calc_manager = nullptr;
    atom_type* atoms;
    int average_steps;
    int steps;
    double dt;

    integrator_type() = delete;
    integrator_type(YAML::Node doc, params_class params);
    void set_binomial_steps(std::mt19937_64 &gen64);
    virtual void integrate() = 0;

    void set_calc_manager(Calc_Manager& calc_manager_ref);
};

#endif