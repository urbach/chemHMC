#ifndef IDENTICAL_PARTICLES_H
#define IDENTICAL_PARTICLES_H

#include <functional>
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>
#include "particles.hpp"



class identical_particles : public particles_type {

public:
    struct cold {};
    struct hot {};
    struct hbTag {};
    struct Tag_potential_all {};
    struct Tag_potential_all_inner_parallel {};
    struct Tag_potential_binning {};
    struct kinetic {};
    struct force {};
    struct Tag_force_inner_parallel {};
    struct Tag_force_binning {};
    struct Tag_quicksort_compare {};
    struct check_in_volume {};
    struct Tag_RDF {};
    struct check_RDF {};
    typedef Kokkos::TeamPolicy<>::member_type  member_type;
    double mass;
    double beta;
    double sbeta;// sqrt(beta)
    double sigma;
    double eps;
    double cutoff;
    const std::string name = "identical_particles";
    std::string name_xyz;

    // constructor
    identical_particles(YAML::Node doc, params_class params);

    double get_beta() { return beta; };
    void print_xyz(params_class params, int traj, double K, double V) override;
    void read_xyz(params_class params) override;
    int how_many_confs_xyz(FILE* file) override;
    void read_next_confs_xyz(FILE* file) override;

    void InitX(params_class params) override;
    void hb() override; // heatbath for the momenta

    std::function<double()>  potential_strategy;
    double potential_all_neighbour();
    double potential_binning();
    double potential_all_neighbour_inner_parallel();
    double compute_potential() override {
        return potential_strategy();
    };
    std::function<double()>  potential_without_binning_strategy;
    double potential_with_binning_set();
    double evaluate_potential() override {
        return potential_without_binning_strategy();
    };

    // function to define the domains
    std::function<void()>  binning_geometry_strategy;
    void cutoff_binning();
    void binning_geometry() override {
        binning_geometry_strategy();
    };

    // binning strategy functions and related
    std::function<void()>  binning_strategy;
    void serial_binning_init();
    void serial_binning();//<-- can be a binning_strategy

    void quick_sort_init();
    int partition(int low, int high);
    int partition_middle(int low, int high);
    int partition_high(int low, int high);
    void quickSort(int low, int high);
    void create_quick_sort();//<-- can be a binning_strategy
    void create_quick_sort_v1();//<-- can be a binning_strategy

    void parallel_binning_init();
    void parallel_binning();//<-- can be a binning_strategy

    void create_binning() override {
        binning_strategy();
    };

    double compute_kinetic_E() override;

    std::function<void()>  force_strategy;
    void compute_force_all();
    void compute_force_all_inner_parallel();
    void compute_force_binning();
    void compute_force() override {
        force_strategy();
    };

    void compute_RDF() override;

    KOKKOS_FUNCTION void operator() (cold, const int i) const;
    KOKKOS_FUNCTION void operator() (hot, const int i) const;
    KOKKOS_FUNCTION void operator() (check_in_volume, const int i) const;


    KOKKOS_FUNCTION void operator() (hbTag, const int i) const;
    // functor to compute the potential
    KOKKOS_FUNCTION void operator() (Tag_potential_all, const int i, double& V) const;
    KOKKOS_FUNCTION void operator() (Tag_potential_all_inner_parallel, const member_type& teamMember, double& V) const;
    KOKKOS_FUNCTION void operator() (Tag_potential_binning, const member_type& teamMember, double& V) const;
    // functor to compute the kinetic energy
    KOKKOS_FUNCTION void operator() (kinetic, const int& i, double& K) const;

    // functor to compute the forces
    KOKKOS_FUNCTION void operator() (force, const int i) const; //declaration of functor
    KOKKOS_FUNCTION void operator() (Tag_force_inner_parallel, const member_type& teamMember) const;
    KOKKOS_FUNCTION void operator() (Tag_force_binning, const member_type& teamMember) const;

    // functor for the RDF
    KOKKOS_FUNCTION void operator() (Tag_RDF, const member_type& teamMember) const; //declaration of functor
    KOKKOS_FUNCTION void operator() (check_RDF, const int& i, double& update) const;

    void compute_coeff_momenta() override;
    void compute_coeff_position() override;
    void update_momenta(const double dt_) override;
    void update_positions(const double dt_) override;
    void print_RDF() override;
    void write_header_RDF(FILE* file, int confs) override;
    virtual void write_RDF(FILE* file, int iconf);


    ~identical_particles() {};
};



// we need a ruduction of 3 double array
template< class ScalarType, int N >
struct array_type {
    ScalarType the_array[N];

    KOKKOS_INLINE_FUNCTION   // Default constructor - Initialize to 0's
        array_type() {
        for (int i = 0; i < N; i++) { the_array[i] = 0; }
    }
    KOKKOS_INLINE_FUNCTION   // Copy Constructor
        array_type(const array_type& rhs) {
        for (int i = 0; i < N; i++) {
            the_array[i] = rhs.the_array[i];
        }
    }
    KOKKOS_INLINE_FUNCTION   // add operator
        array_type& operator += (const array_type& src) {
        for (int i = 0; i < N; i++) {
            the_array[i] += src.the_array[i];
        }
        return *this;
    }
};
typedef array_type<double, dim_space> space_vector;  // used to simplify code below
namespace Kokkos { //reduction identity must be defined in Kokkos namespace
    template<>
    struct reduction_identity< space_vector > {
        KOKKOS_FORCEINLINE_FUNCTION static space_vector sum() {
            return space_vector();
        }
    };
}


#endif