#ifndef INTEGRATOR_H
#define INTEGRATOR_H


#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>
#include "global.hpp"
#include "read_infile.hpp"
#include "particles.hpp"

class integrator_type {

public:
    particles_type* particles;

    integrator_type() {};
    integrator_type(YAML::Node doc);
    virtual void integrate() = 0;
    virtual void update_momenta() = 0;
    virtual void update_position() = 0;
};


class LEAP: public integrator_type {

public:
    LEAP() {};
    LEAP(YAML::Node doc);
    void integrate() override {};
    void update_momenta() override {};
    void update_position() override {};
};
#endif