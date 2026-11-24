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

//////////////////////////////
//   CUTOFF TREATMENT       //
//////////////////////////////

/// How the LJ pair interaction is brought to zero at the cutoff r_c (input key LJ: cutoff_scheme).
///   none        : plain truncation.                  V and F jump at r_c.
///   shift       : V(r) - V(r_c).                     V continuous, F jumps at r_c ("truncated & shifted",
///                                                    same as LAMMPS pair_modify shift / GROMACS Potential-shift).
///   force_shift : V(r) - V(r_c) - (r - r_c) V'(r_c). V and F continuous, V'' jumps at r_c.
///   switch      : V(r) * S(r) with a C^4 polynomial switch S on [switch_on, r_c].
///                 V and its first four derivatives are continuous; V is unmodified below switch_on.
/// A jump in the k-th derivative of V at r_c limits the energy error of ANY integrator to O(dt^k)
/// once pairs cross the cutoff, so force_shift/switch are needed to observe orders > 1 (or > 2).
enum LJ_cutoff_scheme : int { LJ_CUT_NONE = 0, LJ_CUT_SHIFT = 1, LJ_CUT_FORCE_SHIFT = 2, LJ_CUT_SWITCH = 3 };

struct LJ_cutoff {
    int scheme = LJ_CUT_NONE;
    double r_c = 0.0;
    double r_on = 0.0;
    Kokkos::View<double**> v_shift;  ///< V(r_c)/4 for each type pair (the factor 4 is applied after the reduction)
    Kokkos::View<double**> dv_shift; ///< dV/dr at r_c for each type pair

    /// Read cutoff_scheme / switch_on (or the legacy key shift_potential) from the LJ block of the input.
    void parse(YAML::Node lj);
    /// Precompute the per-type-pair constants. Must be called after the cutoff is known.
    void init(const Kokkos::View<double**>::HostMirror& h_epsilon_mat,
              const Kokkos::View<double**>::HostMirror& h_sigma_mat, double cutoff_squared);
    std::string name() const;
};

/// C^4 switching polynomial on x = (r - r_on)/(r_c - r_on) in [0,1]: S = 1 - x^5 (126 - 420x + 540x^2 - 315x^3 + 70x^4).
/// Returns S and writes dS/dr (dS/dx = -630 x^4 (1-x)^4).
KOKKOS_INLINE_FUNCTION double lj_switch(const double r, const double r_on, const double r_c, double& dS_dr) {
    const double w = r_c - r_on;
    const double x = (r - r_on) / w;
    const double x4 = x * x * x * x;
    const double omx = 1.0 - x;
    dS_dr = -630.0 * x4 * omx * omx * omx * omx / w;
    return 1.0 - x4 * x * (126.0 + x * (-420.0 + x * (540.0 + x * (-315.0 + x * 70.0))));
}

/// Pair energy divided by 4, for r2 < r_c^2.
KOKKOS_INLINE_FUNCTION double lj_pair_energy_quarter(const int scheme, const double epsilon, const double sigma,
                                                     const double r2, const double r_c, const double r_on,
                                                     const double v_shift_q, const double dv_c) {
    const double sr2 = sigma * sigma / r2;
    const double sr6 = sr2 * sr2 * sr2;
    double v = epsilon * sr6 * (sr6 - 1.0);
    if (scheme == LJ_CUT_SHIFT) {
        v -= v_shift_q;
    } else if (scheme == LJ_CUT_FORCE_SHIFT) {
        v -= v_shift_q + 0.25 * (Kokkos::sqrt(r2) - r_c) * dv_c;
    } else if (scheme == LJ_CUT_SWITCH) {
        const double r = Kokkos::sqrt(r2);
        if (r > r_on) {
            double dS_dr;
            v *= lj_switch(r, r_on, r_c, dS_dr);
        }
    }
    return v;
}

/// (1/r) dV/dr for r2 < r_c^2. The code stores grad V (not -grad V) in f, so f_i += this * (x_i - x_j).
KOKKOS_INLINE_FUNCTION double lj_pair_dvdr_over_r(const int scheme, const double epsilon, const double sigma,
                                                  const double r2, const double r_c, const double r_on,
                                                  const double dv_c) {
    const double sr2 = sigma * sigma / r2;
    const double sr6 = sr2 * sr2 * sr2;
    double g = 48.0 * epsilon * (sr6 * (-sr6 + 0.5) / r2);
    if (scheme == LJ_CUT_FORCE_SHIFT) {
        g -= dv_c / Kokkos::sqrt(r2);
    } else if (scheme == LJ_CUT_SWITCH) {
        const double r = Kokkos::sqrt(r2);
        if (r > r_on) {
            double dS_dr;
            const double S = lj_switch(r, r_on, r_c, dS_dr);
            const double v = 4.0 * epsilon * sr6 * (sr6 - 1.0);
            g = g * S + v * dS_dr / r;
        }
    }
    return g;
}

//////////////////////////////
//   POTENTIAL CLASSES      //
//////////////////////////////

class LJ : public Calc {
public:
    LJ() = default;
    LJ(YAML::Node doc);
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

    LJ_cutoff cutoff; ///< treatment of the interaction at the cutoff

    double L[dim_space];
    double inverse_L[dim_space];
    double inverse_halved_L[dim_space];
    double cutoff_squared;
};

class LJ_verlet : public Calc {
public:
    LJ_verlet(YAML::Node doc);
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

    LJ_cutoff cutoff; ///< treatment of the interaction at the cutoff

    double L[dim_space];
    double inverse_L[dim_space];
    double inverse_halved_L[dim_space];
    double cutoff_squared;
};
#endif
