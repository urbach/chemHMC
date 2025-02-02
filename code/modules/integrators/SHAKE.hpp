#ifndef SHAKE_HPP
#define SHAKE_HPP

#include "integrator_type.hpp"

class VELOCITY_VERLET_SHAKE : public integrator_type {

public:
    struct Tag_SHAKE {};
    struct Tag_RATTLE {};
    VELOCITY_VERLET_SHAKE() = delete;
    VELOCITY_VERLET_SHAKE(YAML::Node doc, params_class params);
    void integrate() override;
    void apply_SHAKE();  // Position correction
    void apply_RATTLE(); // Velocity correction
};

#endif