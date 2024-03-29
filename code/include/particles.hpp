#ifndef PARTICLES_H
#define PARTICLES_H

#include <functional>
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>
#include "global.hpp"
#include "read_infile.hpp"

typedef Kokkos::View<double*> t_RDF;
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

    type_x  x;
    type_p  p;
    type_f  f;
    bool initHostMirror;
    // the host mirror of x is used to restore the position before the MD in case of a rejection
    type_x::HostMirror h_x;
    type_p::HostMirror h_p;

    int nbin[dim_space], bintot;
    double sizebin[dim_space];
    std::string rng_device_state;
    std::string algorithm;

    t_bincount bincount;
    t_binoffsets binoffsets;
    t_permute_vector permute_vector;
    t_permute_vector permute_vector_temp;

    t_bincount::HostMirror h_bincount;
    t_binoffsets::HostMirror h_binoffsets;
    t_permute_vector::HostMirror h_permute_vector;

    t_bool before;
    t_bool after;

    //RDF
    int NbRDF;
    double LmaxRDF;
    double size_bRDF;
    std::string filename_RDF;
    t_RDF RDF;
    t_RDF::HostMirror h_RDF;

    // rng
    RandPoolType rand_pool;
    using device_type = typename Kokkos::DefaultExecutionSpace::device_type;
    using state_data_type = Kokkos::View<uint64_t**, device_type>;
    int padding;
    typename state_data_type::HostMirror hs;


    void save_device_rng();
    void load_device_rng();
    // constructor
    particles_type(YAML::Node doc, params_class params);

    virtual double get_beta() = 0;
    virtual void InitX(params_class params) = 0;

    void printx();
    void printp();
    virtual void print_xyz(params_class params, int traj, double K, double V) = 0;
    virtual void read_xyz(params_class params) = 0;
    virtual int how_many_confs_xyz(FILE* file) = 0;
    virtual void read_next_confs_xyz(FILE* file) = 0;


    virtual void hb() = 0;
    virtual double compute_potential() = 0;
    virtual double evaluate_potential() = 0;
    virtual double compute_kinetic_E() = 0;
    virtual void compute_force() = 0;
    virtual void compute_coeff_momenta() = 0;
    virtual void compute_coeff_position() = 0;
    virtual void update_momenta(const double dt_) = 0;
    virtual void update_positions(const double dt_) = 0;
    virtual void binning_geometry() = 0;
    virtual void create_binning() = 0;
    virtual void compute_RDF() = 0;
    virtual void print_RDF() = 0;
    virtual void write_header_RDF(FILE* file, int confs) = 0;
    virtual void write_RDF(FILE* file, int iconf) = 0;

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