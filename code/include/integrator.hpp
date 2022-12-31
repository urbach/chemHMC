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
    int steps;
    double dt;
    int N;

    integrator_type() {};
    integrator_type(YAML::Node doc);
    virtual void integrate() = 0;
    virtual void update_momenta(const double dt_) = 0;
    virtual void update_positions(const double dt_) = 0;
};


class LEAP: public integrator_type {

public:
    LEAP() {};
    LEAP(YAML::Node doc);
    void integrate() override;
    void update_momenta(const double dt_) override ;
    void update_positions(const double dt_) override;
};
#endif