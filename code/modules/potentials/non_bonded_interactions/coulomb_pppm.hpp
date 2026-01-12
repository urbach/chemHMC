#ifndef COULOMB_PPPM_HPP
#define COULOMB_PPPM_HPP

#include "global.hpp"
#include "particles.hpp"

class Coulomb_pppm : public Calc {
public:
    // Ewald sum
    double ewald_alpha; // width of the gaussians
    double r_c; // real-space cutoff for the ewald sum
    double r_c2;
    double realspace_cutoff;
    double realspace_cutoff_squared;

    double ewald_accuracy; // rms accuracy of the ewald sum
    int k_max;
    double V_self = 0.0; // self interaction energy in the ewald sum is precomputed and stored.
    Kokkos::View<double*> charge;
    Kokkos::View<double*>::HostMirror h_charge;

    void init(const particles_instance& particles) override;
    double potential(const particles_instance& particles) override;
    void force(const particles_instance& particles, type_f& f) override;

    Coulomb_pppm(YAML::Node doc, params_class& params);

    ////////////////////////////////////////////////////////////////////////////////////

    double chargesum;
    double chargesquaredsum;

    double grid_spacing[3];
    int grid_size[3];
    int stencil_order = 5;

    double kspace_base[3]; // basis vectors in kspace

    // Views
    Kokkos::View<double***> density;
    Kokkos::View<double***>::HostMirror h_density;
    // reciprocal‐space force components after inverse‐FFT
    Kokkos::View<double***> vdx_brick;
    Kokkos::View<double***>::HostMirror h_vdx_brick;
    Kokkos::View<double***> vdy_brick;
    Kokkos::View<double***>::HostMirror h_vdy_brick;
    Kokkos::View<double***> vdz_brick;
    Kokkos::View<double***>::HostMirror h_vdz_brick;

    // FFT buffers
    Kokkos::View<Kokkos::complex<double>> density_fft;
    Kokkos::View<Kokkos::complex<double>>::HostMirror h_density_fft;
    Kokkos::View<double*> greensfn;
    Kokkos::View<double*>::HostMirror h_greensfn;
    Kokkos::View<Kokkos::complex<double>> work1;
    Kokkos::View<Kokkos::complex<double>>::HostMirror h_work1;
    Kokkos::View<Kokkos::complex<double>> work2;
    Kokkos::View<Kokkos::complex<double>>::HostMirror h_work2;

    // k-vector tables
    Kokkos::View<double*> fkx;
    Kokkos::View<double*>::HostMirror h_fkx;
    Kokkos::View<double*> fky;
    Kokkos::View<double*>::HostMirror h_fky;
    Kokkos::View<double*> fkz;
    Kokkos::View<double*>::HostMirror h_fkz;

    // spline interpolation tables
    Kokkos::View<double*> greensfn_fac;
    Kokkos::View<double*>::HostMirror h_greensfn_fac;
    Kokkos::View<double**> rho_coeff;
    Kokkos::View<double**>::HostMirror h_rho_coeff;
    Kokkos::View<double**> drho_coeff;
    Kokkos::View<double**>::HostMirror h_drho_coeff;
    Kokkos::View<double**> rho1d;
    Kokkos::View<double**>::HostMirror h_rho1d;
    Kokkos::View<double**> drho1d;
    Kokkos::View<double**>::HostMirror h_drho1d;

    // Init
    void estimate_alpha(int N, double V);
    void init_grid(const particles_instance& particles, double L[3]);
    double kspace_error_1D(double grid_spacing, double box_length, int N);
    double total_kspace_error(const particles_instance& particles);
    void tune_alpha(const particles_instance& particles);
    void allocate_views(const particles_instance& particles);
    void compute_greensfn_fac();
    void compute_rho_fac();
    void compute_greensfn_mesh(double L[3]);
    double greensfn_denominator(double snx, double sny, double snz);

    // NR solver functions
    double rspace_kspace_diff(const particles_instance& particles);
    double num_derivative_diff(const particles_instance& particles);


    // Coefficients for kspace error estimation from 
    // Deserno, M.; Holm, C.; J. Chem. Phys. 109, 7694–7701 (1998)
    double error_coeffs[6][5];
};

#endif