#ifndef IDENTICAL_PARTICLES_H
#define IDENTICAL_PARTICLES_H

#include <functional>
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>
#include "particles.hpp"



class identical_particles: public particles_type {

public:
    struct cold {};
    struct hot {};
    struct hbTag {};
    struct Tag_potential_all {};
    struct Tag_potential_binning {};
    struct kinetic {};
    struct force {};
    struct Tag_quicksort_compare {};
    typedef Kokkos::TeamPolicy<>::member_type  member_type;
    double mass;
    double beta;
    double sbeta;// sqrt(beta)
    double sigma;
    double eps;
    double cutoff;
    const std::string name = "identical_particles";

    // constructor
    identical_particles(YAML::Node doc);

    void InitX() override;
    void hb() override; // heatbath for the momenta

    std::function<double()>  potential_strategy;
    double potential_all_neighbour();
    double potential_binning();
    double compute_potential() override {
        return potential_strategy();
    };

    std::function<void()>  binning_geometry_strategy;
    void cutoff_binning();
    void binning_geometry() override {
        return binning_geometry_strategy();
    };

    std::function<void()>  binning_strategy;
    void serial_binning_init();
    void serial_binning();

    void quick_sort_init();
    // KOKKOS_FUNCTION bool compare(int j, int pi);
    int partition(int low, int high);
    int partition_middle(int low, int high);
    int partition_high(int low, int high);
    // KOKKOS_FUNCTION void operator() (Tag_quicksort_compare,  const int j, int& update) const;
    void quickSort(int low, int high);
    void create_quick_sort();

    void create_binning() override {
        return binning_strategy();
    };

    double compute_kinetic_E() override;
    void compute_force() override;  //declaration of the function

    KOKKOS_FUNCTION void operator() (cold, const int i) const;
    KOKKOS_FUNCTION void operator() (hot, const int i) const;

    KOKKOS_FUNCTION void operator() (hbTag, const int i) const;
    // functor to compute the potential
    KOKKOS_FUNCTION void operator() (Tag_potential_all, const int i, double& V) const;
    KOKKOS_FUNCTION void operator() (Tag_potential_binning, const member_type& teamMember, double& V) const;
    // functor to compute the kinetic energy
    KOKKOS_FUNCTION void operator() (kinetic, const int i, double& K) const;


    // functor to compute the forces
    KOKKOS_FUNCTION void operator() (force, const int i) const; //declaration of functor
    ~identical_particles() {};

    void compute_coeff_momenta() override;
    void compute_coeff_position() override;
    void update_momenta(const double dt_) override;
    void update_positions(const double dt_) override;
};




#endif