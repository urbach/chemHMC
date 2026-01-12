#ifndef NEIGHBOR_LIST_HPP
#define NEIGHBOR_LIST_HPP

#include <Kokkos_Core.hpp>
#include "yaml-cpp/yaml.h"
#include "global.hpp"

class particles_instance; // We need to declare this here because we cant include particles.hpp directly

class Neighbor_list {
public:
    Kokkos::View<int*> neighbour_count;
    Kokkos::View<int*>::HostMirror h_neighbour_count;
    Kokkos::View<int**> verlet_list;
    Kokkos::View<int**>::HostMirror h_verlet_list;
    Kokkos::View<double*[3]> x_last;
    Kokkos::View<double*[3]>::HostMirror h_x_last;
    Kokkos::View<double*> disp2;
    Kokkos::View<double*>::HostMirror h_disp2;

    int update_every;
    int moves_since_last_update = 0;
    bool list_used = false; // is set to false on each build, can be set to true by user

    double time_list_build = 0.0; // Time for neighbor list builds

    double neighbor_cutoff;
    double neighbor_cutoff_squared;
    double skin_distance_squared; // buffer between cutoff and max neighbor distance

    void init_verlet_list(YAML::Node& doc, particles_instance& particles);
    struct Tag_build_verlet_list {};
    virtual void build_verlet_list(particles_instance& particles);
    void build(particles_instance& particles);

    virtual ~Neighbor_list() = default;
};

class Neighbor_list_bonds : public Neighbor_list {
public:
    struct Tag_verlet_remove_bonds {};
    void build_verlet_list(particles_instance& particles) override;
    void build_initial_verlet_list(particles_instance& particles);
    void remove_bonds(particles_instance& particles);
};

class Neighbor_list_cell : public Neighbor_list {
public:
    void build_verlet_list(particles_instance& particles) override;
    void build_initial_cell_list(particles_instance& particles);
    void init_verlet_list(YAML::Node& doc, particles_instance& particles);
};

#endif