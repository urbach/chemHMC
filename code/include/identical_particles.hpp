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
    struct Tag_potential_binning {};
    struct kinetic {};
    struct force {};
    struct Tag_force_binning {};
    struct Tag_quicksort_compare {};
    struct check_in_volume {};
    struct Tag_RDF {};
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
    identical_particles(YAML::Node doc);

    double get_beta() { return beta; };
    void print_xyz(int traj, double K, double V) override;
    void read_xyz() override;
    int how_many_confs_xyz(FILE* file) override;
    void read_next_confs_xyz(FILE* file) override;
    
    void InitX() override;
    void hb() override; // heatbath for the momenta

    std::function<double()>  potential_strategy;
    double potential_all_neighbour();
    double potential_binning();
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
    KOKKOS_FUNCTION void operator() (Tag_potential_binning, const member_type& teamMember, double& V) const;
    // functor to compute the kinetic energy
    KOKKOS_FUNCTION void operator() (kinetic, const int i, double& K) const;

    // functor to compute the forces
    KOKKOS_FUNCTION void operator() (force, const int i) const; //declaration of functor
    KOKKOS_FUNCTION void operator() (Tag_force_binning, const member_type& teamMember) const;

    // functor for the RDF
    KOKKOS_FUNCTION void operator() (Tag_RDF, const member_type& teamMember) const; //declaration of functor

    void compute_coeff_momenta() override;
    void compute_coeff_position() override;
    void update_momenta(const double dt_) override;
    void update_positions(const double dt_) override;
    void print_RDF() override;
    void write_header_RDF(FILE* file, int confs) override;
    virtual void write_RDF(FILE* file, int iconf);


    ~identical_particles() {};
};



#endif