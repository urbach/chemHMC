#ifndef INTEGRATOR_H
#define INTEGRATOR_H


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

class VELOCITY_VERLET : public integrator_type {

public:
    VELOCITY_VERLET() = delete;
    VELOCITY_VERLET(YAML::Node doc, params_class params);
    void integrate() override;
};

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

class LEAP : public integrator_type {

public:
    LEAP() = delete;
    LEAP(YAML::Node doc, params_class params);
    void integrate() override;

};

class OMF2 : public integrator_type {

public:
    const double lambda;
    const double oneminus2lambda;
    OMF2() = delete;
    OMF2(YAML::Node doc, params_class params);
    void integrate() override;

};



class OMF4 : public integrator_type {

public:
    const double rho;
    const double theta;
    const double vartheta;
    const double lambda;
    const double dtau;
    const double eps[10];
    OMF4() = delete;
    OMF4(YAML::Node doc, params_class params);
    void integrate() override;

};

#endif