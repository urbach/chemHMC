#ifndef PARTICLES_H
#define PARTICLES_H

#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>
#include "global.hpp"
#include "read_infile.hpp"

class particles_type { 

public:

    int N;

    double coeff_p;
    double coeff_x;

    type_x  x;
    type_x  x_old;
    type_p  p;
    type_f  f;
    bool initHostMirror;
    type_x::HostMirror h_x;
    type_p::HostMirror h_p;

    params_class params;

    RandPoolType rand_pool;
    // constructor
    particles_type(YAML::Node doc);

    virtual void InitX() = 0;

    void printx();
    void printp();

    virtual void hb() = 0;
    virtual double compute_potential() = 0;
    virtual double compute_kinetic_E() = 0;
    virtual void compute_force() = 0;
    virtual void compute_coeff_momenta() = 0;
    virtual void compute_coeff_position() = 0;
    virtual void update_momenta(const double dt_) = 0;
    virtual void update_positions(const double dt_) = 0;
};

class identical_particles: public particles_type {

public:
    struct cold {};
    struct hot {};
    struct hbTag {};
    struct potential {};
    struct kinetic {};
    struct force {};
    double mass;
    double beta;
    double sbeta;
    double sigma;
    double eps;
    double cutoff;
    const std::string name = "identical_particles";
    
    // constructor
    identical_particles(YAML::Node doc);

    void InitX() override;
    void hb()override;
    double compute_potential() override;
    double compute_kinetic_E() override;
    void compute_force() override;  //declaration of the function

    KOKKOS_FUNCTION void operator() (cold, const int i) const;
    KOKKOS_FUNCTION void operator() (hot, const int i) const;

    KOKKOS_FUNCTION void operator() (hbTag, const int i) const;
    // functor to compute the potential
    KOKKOS_FUNCTION void operator() (potential, const int i, double& V) const;
    // functor to compute the kinetic energy
    KOKKOS_FUNCTION void operator() (kinetic, const int i, double& K) const;


    // functor to compute the forces
    KOKKOS_FUNCTION void operator() (force, const int i) const; //declaration of functor
    ~identical_particles() {};

    void compute_coeff_momenta() override;
    void compute_coeff_position() override;
     void update_momenta(const double dt_) override ;
    void update_positions(const double dt_) override;
};



#endif