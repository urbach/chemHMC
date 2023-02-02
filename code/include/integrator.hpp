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

    integrator_type() = delete;
    integrator_type(YAML::Node doc);
    virtual void integrate() = 0;
    
};


class LEAP: public integrator_type {

public:
    LEAP() = delete;
    LEAP(YAML::Node doc);
    void integrate() override;
   
};

class OMF2: public integrator_type {

public:
    const double lambda;
    const double oneminus2lambda ;
    OMF2() = delete;
    OMF2(YAML::Node doc);
    void integrate() override;

};

#endif