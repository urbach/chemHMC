#ifndef COULOMB_HPP
#define COULOMB_HPP

#include "global.hpp"
#include "particles.hpp"

class Coulomb : public Calc {
public:
    // Ewald sum
    double ewald_alpha; // width of the gaussians
    double sqrt_ewald_alpha;
    double r_c; // real-space cutoff for the ewald sum
    double r_c2;
    double ewald_accuracy; // rms accuracy of the ewald sum
    int k_max;
    double V_self = 0.0; // self interaction energy in the ewald sum is precomputed and stored.
    Kokkos::View<double*> charge;
    Kokkos::View<double*>::HostMirror h_charge;

    void init(const particles_instance& particles) override;
    double potential(const particles_instance& particles) override;
    void force(const particles_instance& particles, type_f& f) override;

    struct Tag_potential_ewald_real {};
    struct Tag_potential_ewald_reciprocal {};
    struct Tag_potential_ewald_self {};
    double compute_ewald_real(const particles_instance& particles);
    double compute_ewald_reciprocal(const particles_instance& particles);
    double compute_ewald_self(const particles_instance& particles);

    struct Tag_force_ewald_real {};
    struct Tag_force_ewald_reciprocal {};
    void compute_ewald_real_forces(const particles_instance& particles,type_f& f);
    void compute_ewald_reciprocal_forces(const particles_instance& particles,type_f& f);
};

#endif