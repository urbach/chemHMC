#ifndef VELOCITY_VERLET_HPP
#define VELOCITY_VERLET_HPP

#include "integrator_type.hpp"

class VELOCITY_VERLET : public integrator_type {

public:
    VELOCITY_VERLET() = delete;
    VELOCITY_VERLET(YAML::Node doc, params_class params);
    void integrate() override;
};

#endif