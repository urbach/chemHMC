#ifndef PARTICLES_H
#define PARTICLES_H

#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>
#include "global.hpp"
#include "read_infile.hpp"

class particles_type { // Just the thing that holds all variables

public:

    int N;

    params_class params;
    type_x  x;
    type_p  p;
    type_f  f;
    bool initHostMirror;
    type_x::HostMirror h_x;
    type_p::HostMirror h_p;

    RandPoolType rand_pool;
    // constructor
    particles_type(YAML::Node doc);

    virtual void InitX() = 0;

    void printx();
    void printp();

    virtual void hb() = 0;
    virtual double compute_potential() = 0;
    virtual void compute_force() = 0;
};

class identical_particles: public particles_type {

public:
    struct cold {};
    struct hot {};
    struct hbTag {};
    struct potential {};
    struct force {};
    const std::string name = "identical_particles";
    double mass;
    double beta;
    double sbeta;
    double sigma;
    double eps;
    double cutoff;
    // constructor
    identical_particles(YAML::Node doc);

    void InitX() override;
    void hb()override;
    double compute_potential() override;
    void compute_force() override;  //declaration of the function

    KOKKOS_FUNCTION void operator() (cold, const int i) const;
    KOKKOS_FUNCTION void operator() (hot, const int i) const;

    KOKKOS_FUNCTION void operator() (hbTag, const int i) const;
    // functor to compute the potential
    KOKKOS_FUNCTION void operator() (potential, const int i, double& V) const;

    // functor to compute the forces
    KOKKOS_FUNCTION void operator() (force, const int i) const; //declaration of functor
    ~identical_particles() {};
};



#endif