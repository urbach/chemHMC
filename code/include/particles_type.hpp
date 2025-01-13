#ifndef PARTICLES_TYPE_H
#define PARTICLES_TYPE_H

#include <functional>
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>
#include "global.hpp"
#include "read_infile.hpp"

typedef Kokkos::View<int*> t_bincount;
typedef Kokkos::View<int*> t_binoffsets;
typedef Kokkos::View<int*> t_permute_vector;
typedef Kokkos::View<bool*> t_bool;
typedef Kokkos::View<int*> t_prefix;

class particles_type {

public:

    int N;
    int seed;
    double coeff_p;
    double coeff_x;
    double L[dim_space];
    double inverse_L[dim_space];
    double inverse_halved_L[dim_space];

    type_x  x;      ///< Kokkos view containing the positions
    type_p  p;      ///< Kokkos view containing the momenta
    type_f  f;      ///< Kokkos view containing the forces
    type_id id;     ///< Kokkos view containing the type_ids 
    // the host mirror of x is used to restore the position before the MD in case of a rejection
    type_x::HostMirror h_x;     ///< Host mirror of x (positions).
    type_p::HostMirror h_p;     ///< Host_mirror of p (momenta)

    int nbin[dim_space], bintot;
    double sizebin[dim_space];
    std::string rng_device_state;
    std::string algorithm;
    std::vector<std::string> label_xyz;

    t_bincount bincount;
    t_binoffsets binoffsets;
    t_permute_vector permute_vector;
    t_permute_vector permute_vector_temp;

    t_bincount::HostMirror h_bincount;
    t_binoffsets::HostMirror h_binoffsets;
    t_permute_vector::HostMirror h_permute_vector;

    t_bool before;
    t_bool after;

    // rng
    RandPoolType rand_pool;

    // constructor
    particles_type();

    virtual double get_beta() = 0;
    virtual void InitX() = 0;

    virtual void print_xyz(params_class params, int traj, double K, double V) = 0;

    virtual void hb() = 0;
    virtual double compute_potential() = 0;
    virtual double evaluate_potential() = 0;
    virtual double compute_kinetic_E() = 0;
    virtual void compute_force() = 0;
    virtual void compute_coeff_momenta() = 0;
    virtual void compute_coeff_position() = 0;
    virtual void update_momenta(const double dt_) = 0;
    virtual void update_positions(const double dt_) = 0;
    virtual void build_verlet_list() = 0;
    virtual void build_bondless_verlet_list() = 0;
    virtual void minimize_energy(YAML::Node& doc) = 0;
    KOKKOS_INLINE_FUNCTION void lextoc(int ib, int& bx, int& by, int& bz) const {
        bz = ib / (nbin[0] * nbin[1]);
        by = (ib - bz * nbin[0] * nbin[1]) / (nbin[0]);
        bx = ib - nbin[0] * (by + bz * nbin[1]);
    };
    KOKKOS_INLINE_FUNCTION int ctolex(int& bx, int& by, int& bz)  const {
        return bx + nbin[0] * (by + bz * nbin[1]);
    };
    KOKKOS_INLINE_FUNCTION int which_bin(type_x  x, int i) const {
        int bx = floor(x(i, 0) / sizebin[0]);
        int by = floor(x(i, 1) / sizebin[1]);
        int bz = floor(x(i, 2) / sizebin[2]);
        return ctolex(bx, by, bz);
    };
};

#endif