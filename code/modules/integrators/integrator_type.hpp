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
    int steps;
    double dt;

    integrator_type() = delete;
    integrator_type(YAML::Node doc, params_class params);
    virtual void integrate() = 0;

    void set_calc_manager(Calc_Manager& calc_manager_ref);

protected:
    /// Must be called after every position update: rebuilds the neighbour list if atoms moved too far.
    void update_neighbor_list() {
        if (particles->neighbor_list_used) {
            particles->neighbor_list->build_verlet_list(*particles);
        }
    }
};

#endif