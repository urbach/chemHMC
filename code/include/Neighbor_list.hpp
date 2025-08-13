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

    int update_every;
    int moves_since_last_update = 0;

    double time_list_build = 0.0; // Time for neighbor list builds

    double neighbor_cutoff;
    double neighbor_cutoff_squared;

    void init_verlet_list(YAML::Node& doc, particles_instance& particles);
    struct Tag_build_verlet_list {};
    virtual void build_verlet_list(particles_instance& particles);

    virtual ~Neighbor_list() = default;
};

class Neighbor_list_bonds : public Neighbor_list {
public:
    struct Tag_build_verlet_list {};
    struct Tag_verlet_remove_bonds {};
    void build_verlet_list(particles_instance& particles) override;
    void build_initial_verlet_list(particles_instance& particles);
    void remove_bonds(particles_instance& particles);
};

#endif