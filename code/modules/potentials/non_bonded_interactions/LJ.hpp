#ifndef LJ_HPP
#define LJ_HPP

#include "Calc.hpp"
#include "global.hpp"
#include <Kokkos_Core.hpp>
#include <functional>
#include <vector>
#include <atom.hpp>
#include <iostream>
#include <iomanip>
#include "yaml-cpp/yaml.h"
#include "particles.hpp"
class LJ : public Calc {
public:
    LJ() = default;
    ~LJ() override = default;

    void init(const particles_instance& particles) override;
    double potential(const particles_instance& particles) override;
    void force(const particles_instance& particles, type_f& f) override;

    struct Tag_potential_AMIC_inner_parallel {};
    struct Tag_force_AMIC_inner_parallel {};

private:
    typedef Kokkos::TeamPolicy<>::member_type  member_type;
    Kokkos::View<double**> epsilon_mat; ///< matrix containing the LJ_epsilon parameters
    Kokkos::View<double**> sigma_mat; ///< matrix containing the LJ_sigma parameters
    Kokkos::View<double**>::HostMirror h_epsilon_mat; ///< host mirror of epsilon_mat
    Kokkos::View<double**>::HostMirror h_sigma_mat; ///< host mirror of sigma_mat

    double L[dim_space];
    double inverse_L[dim_space];
    double inverse_halved_L[dim_space];
    double cutoff_squared;
};

class LJ_verlet : public Calc {
public:
    LJ_verlet() = default;
    ~LJ_verlet() override = default;

    void init(const particles_instance& particles) override;
    double potential(const particles_instance& particles) override;
    void force(const particles_instance& particles, type_f& f) override;

    struct Tag_potential_verlet {};
    struct Tag_force_verlet {};

private:
    typedef Kokkos::TeamPolicy<>::member_type  member_type;
    Kokkos::View<double**> epsilon_mat; ///< matrix containing the LJ_epsilon parameters
    Kokkos::View<double**> sigma_mat; ///< matrix containing the LJ_sigma parameters
    Kokkos::View<double**>::HostMirror h_epsilon_mat; ///< host mirror of epsilon_mat
    Kokkos::View<double**>::HostMirror h_sigma_mat; ///< host mirror of sigma_mat

    double L[dim_space];
    double inverse_L[dim_space];
    double inverse_halved_L[dim_space];
    double cutoff_squared;
};
#endif
