#ifndef LEAPFROG_HPP
#define LEAPFROG_HPP

#include "integrator_type.hpp"

class LEAP : public integrator_type {
public:
    LEAP() = delete;
    LEAP(YAML::Node doc, params_class params) : integrator_type(doc, params) {}
    void integrate() override;
};

#endif