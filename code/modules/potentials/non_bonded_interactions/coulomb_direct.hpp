#ifndef COULOMB_DIRECT_HPP
#define COULOMB_DIRECT_HPP

#include "global.hpp"
#include "particles.hpp"

class Coulomb_direct : public Calc {
public:
    Kokkos::View<double*> charge;
    Kokkos::View<double*>::HostMirror h_charge;

    double r_c;
    double r_c2;

    void init(const particles_instance& particles) override;
    double potential(const particles_instance& particles) override;
    void force(const particles_instance& particles, type_f& f) override;

    double Coulomb_potential_in_cell_0(const particles_instance& particles);
    double Coulomb_potential_in_cell_k(const particles_instance& particles, int k_x, int k_y, int k_z);

    struct Tag_potential_direct {};
    struct Tag_force_direct {};
};

#endif