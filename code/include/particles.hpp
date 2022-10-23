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
    
    Kokkos::View<double* [dim_space]>  x;
    Kokkos::View<double* [dim_space]>::HostMirror h_x;
    Kokkos::View<double* [dim_space]>  p;
    Kokkos::View<double* [dim_space]>::HostMirror h_p;

    RandPoolType rand_pool;
    void InitX(params_class params);
    KOKKOS_INLINE_FUNCTION
        void operator() (cold, const int& i) const;
    // KOKKOS_INLINE_FUNCTION void operator() (hot, const int i) const;
    // virtual void hb() = 0; // virtual heat-bath routine

};

class identical_particles : public particles_type {

public:
    struct hbTag {};
    const std::string name = "identical_particles";
    identical_particles(YAML::Node doc);
    void hb();
    // KOKKOS_FUNCTION void operator() (hbTag, const int i) const;
    ~identical_particles() {};
};



#endif