#ifndef PARTICLES_H
#define PARTICLES_H

#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>
#include "global.hpp"
#include "read_infile.hpp"

class particles_type { // Just the thing that holds all variables

public:
    struct cold {};
    struct hot {};
    int N;
    double mass;
    double beta;
    params_class params;
    Kokkos::View<double* [dim_space]>  x;
    Kokkos::View<double* [dim_space]>  p;
    bool initHostMirror;
    Kokkos::View<double* [dim_space]>::HostMirror h_x;
    Kokkos::View<double* [dim_space]>::HostMirror h_p;

    RandPoolType rand_pool;
    // constructor
    particles_type(YAML::Node doc, params_class params_to_copy);

    void InitX();

    KOKKOS_INLINE_FUNCTION
        void operator() (cold, const int& i) const;

    void printx();
    void printp();

    virtual void hb() = 0;

};

class identical_particles : public particles_type {

public:
    struct hbTag {};
    const std::string name = "identical_particles";
    // constructor
    identical_particles(YAML::Node doc, params_class params_to_copy);
    void hb()override;
    ~identical_particles() {};
};



#endif