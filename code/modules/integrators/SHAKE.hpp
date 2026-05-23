#ifndef SHAKE_HPP
#define SHAKE_HPP

#include "integrator_type.hpp"
#include <Kokkos_Core.hpp>

class VELOCITY_VERLET_SHAKE : public integrator_type {

public:
    int max_iter;
    double tolerance;
    Kokkos::View<int*> size_1_clusters;
    Kokkos::View<int* [2]> size_2_clusters;
    Kokkos::View<int* [3]> size_3_clusters;
    type_x trial_positions;
    type_x old_positions;
    type_p trial_momenta;

    struct Tag_SHAKE {};
    struct Tag_RATTLE {};
    VELOCITY_VERLET_SHAKE() = delete;
    VELOCITY_VERLET_SHAKE(YAML::Node doc, params_class params);
    void integrate() override;
    void find_shake_clusters();
    void generate_trial_positions();
    void apply_SHAKE();  // Position correction
    void SHAKE_size_1_cluster();
    void SHAKE_size_2_cluster();
    void SHAKE_size_3_cluster();
    void apply_RATTLE(); // Velocity correction
    void generate_trial_momenta();
    void RATTLE_size_1_cluster();
    void RATTLE_size_2_cluster();
    void RATTLE_size_3_cluster();
};

#endif
