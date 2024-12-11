#include "global.hpp"
#include "particles.hpp"



// EWALD SUM
void particles_instance::init_ewald_sum(YAML::Node& doc) {

    // ratio of timing for real/imaginary parts, averaged over 100 runs
    double timer_ratio = 0.046304/0.018513; 
    ewald_alpha = pow(timer_ratio*(M_PI*M_PI*M_PI*N)/(pow(L[0],6)),(1.0/3.0));
    sqrt_ewald_alpha = sqrt(ewald_alpha);
    r_c = check_and_assign_value<double>(doc["ewald"], "cutoff");
    r_c2 = r_c*r_c;
    ewald_accuracy = check_and_assign_value<double>(doc["ewald"], "accuracy");
    k_max = check_and_assign_value<int>(doc["ewald"], "k_max");

    printf("cutoff: %f \n", r_c);
    printf("accuracy: %f \n", ewald_accuracy);
    printf("alpha: %f \n", ewald_alpha);


    charge = Kokkos::View<double*>("charge", h_atom_type_list.extent(0));
    h_charge = Kokkos::create_mirror_view(charge);

    for (int i = 0; i < h_atom_type_list.extent(0); i++) {
        h_charge(i) = h_atom_type_list(i).charge;
    }

    Kokkos::deep_copy(charge, h_charge);
}

double particles_instance::potential_ewald_sum() {
    const double conversion_factor = 332.062934*kcaltointernal; //conversion to amu* A^2/fs^2
    double V_real = compute_ewald_real();
    double V_reciprocal = compute_ewald_reciprocal();
    //self interaction only needs to be computed once
    if (V_self == 0.0) {
        V_self = compute_ewald_self();
    }

    return conversion_factor*(V_real + V_reciprocal + V_self);
}

double particles_instance::compute_ewald_real() {
    double result = 0.0;
    Kokkos::parallel_reduce("ewald-real-space",
        Kokkos::TeamPolicy<Tag_potential_ewald_real>(N, Kokkos::AUTO), *this, result);
    return 0.5 * result;
}

KOKKOS_FUNCTION
void particles_instance::operator() (Tag_potential_ewald_real, const member_type& teamMember, double& V) const {
    const int i = teamMember.league_rank();
    double tmpV = 0.0;
    double sqrt_alpha = sqrt_ewald_alpha;
    double qi = charge(id(i)-1);

    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, N), [=](const int j, double& innerV) {
        if (i != j) {
            double qj = charge(id(j)-1);
            double rij[3];

            // Calculate distance vector and apply periodic boundary conditions
            rij[0] = x(i, 0) - x(j, 0);
            rij[1] = x(i, 1) - x(j, 1);
            rij[2] = x(i, 2) - x(j, 2);

            rij[0] -= int(rij[0] * inverse_halved_L[0]) * L[0];
            rij[1] -= int(rij[1] * inverse_halved_L[1]) * L[1];
            rij[2] -= int(rij[2] * inverse_halved_L[2]) * L[2];

            double r2 = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];
            if (r2 <= r_c2) {
            double r = sqrt(r2);
            innerV += qi * qj * erfc(sqrt_alpha * r) / r;
            }
        }
    }, tmpV);

    Kokkos::single(Kokkos::PerTeam(teamMember), [&]() {
        V += tmpV;
    });
}

double particles_instance::compute_ewald_reciprocal() {
    double result = 0.0;
    int num_kpoints = (2 * k_max + 1) * (2 * k_max + 1) * (2 * k_max + 1); // Total number of k-points

    Kokkos::parallel_reduce("ewald-reciprocal-space",
        Kokkos::TeamPolicy<Tag_potential_ewald_reciprocal>(num_kpoints, Kokkos::AUTO), *this, result);

    return 2 * M_PI * result / (L[0] * L[1] * L[2]); // Normalizing by the volume of the box
}

KOKKOS_FUNCTION
void particles_instance::operator()(Tag_potential_ewald_reciprocal, const member_type& teamMember, double& V) const {
    const int league_rank = teamMember.league_rank();
    const int num_kpoints_2D = (2 * k_max + 1) * (2 * k_max + 1);
    double alpha = ewald_alpha;

    const int kz = league_rank / num_kpoints_2D - k_max;
    const int ky = (league_rank % num_kpoints_2D) / (2 * k_max + 1) - k_max;
    const int kx = league_rank % (2 * k_max + 1) - k_max;

    if (kx == 0 && ky == 0 && kz == 0) return; // Skip the k=0 term

    double tmpV = 0.0;
    double kx_real = 2 * M_PI * kx / L[0];
    double ky_real = 2 * M_PI * ky / L[1];
    double kz_real = 2 * M_PI * kz / L[2];
    double k2 = kx_real * kx_real + ky_real * ky_real + kz_real * kz_real;

    double S_re = 0.0, S_im = 0.0;

    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, N), [=](const int j, double& inner_re, double& inner_im) {
        double kr = kx_real * x(j, 0) + ky_real * x(j, 1) + kz_real * x(j, 2);
        inner_re += charge(id[j]-1) * cos(kr);
        inner_im += charge(id[j]-1) * sin(kr);
    }, Kokkos::Sum<double>(S_re), Kokkos::Sum<double>(S_im));

    double exp_factor = exp(-k2 / (4 * alpha));
    tmpV += (S_re * S_re + S_im * S_im) * exp_factor / k2;

    Kokkos::single(Kokkos::PerTeam(teamMember), [&]() {
        V += tmpV;
    });
}


double particles_instance::compute_ewald_self() {
    double result = 0.0;
    Kokkos::parallel_reduce("ewald-self-energy",
        Kokkos::TeamPolicy<Tag_potential_ewald_self>(N, Kokkos::AUTO), *this, result);
    
    return -sqrt(ewald_alpha/M_PI) * result; // The negative sign accounts for self-interaction  
}

KOKKOS_FUNCTION
void particles_instance::operator()(Tag_potential_ewald_self, const member_type& teamMember, double& V) const {
    const int i = teamMember.league_rank();

    double qi = charge(id(i)-1);
    double self_energy_contribution = qi * qi;

    Kokkos::single(Kokkos::PerTeam(teamMember), [&]() {
        V += self_energy_contribution;
    });
}

void particles_instance::compute_force_ewald() {
    Kokkos::deep_copy(f, 0.0); // Set force to 0
    compute_ewald_real_forces();
    compute_ewald_reciprocal_forces();
}

void particles_instance::compute_ewald_real_forces() {
    Kokkos::parallel_for("ewald-real-space-force",
        Kokkos::TeamPolicy<Tag_force_ewald_real>(N, Kokkos::AUTO), *this);
}

KOKKOS_FUNCTION
void particles_instance::operator()(Tag_force_ewald_real, const member_type& teamMember) const {
    const int i = teamMember.league_rank();
    double alpha = ewald_alpha;
    double sqrt_alpha = sqrt_ewald_alpha;
    double qi = charge(id(i)-1);

    // Temporary force accumulators for atom i
    double fx_i = 0.0;
    double fy_i = 0.0;
    double fz_i = 0.0;

    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, N), [=](const int j, double& fx_tmp, double& fy_tmp, double& fz_tmp) {
        if (i != j) {
            double qj = charge(id(j)-1);
            double rij[3];

            // Calculate distance vector and apply minimum image convention
            rij[0] = x(i, 0) - x(j, 0);
            rij[1] = x(i, 1) - x(j, 1);
            rij[2] = x(i, 2) - x(j, 2);

            // Apply periodic boundary conditions
            rij[0] -= round(rij[0] * inverse_L[0]) * L[0];
            rij[1] -= round(rij[1] * inverse_L[1]) * L[1];
            rij[2] -= round(rij[2] * inverse_L[2]) * L[2];

            double r2 = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];

            if (r2 < r_c2) {
                double r = sqrt(r2);
                double erfc_alpha_r = erfc(sqrt_alpha * r);
                double exp_alpha2_r2 = exp(-alpha * r2);

                double force_prefactor = qi * qj * (2.0 * sqrt_alpha / sqrt(M_PI) * exp_alpha2_r2 + erfc_alpha_r / r) / (r2);

                // Accumulate forces
                fx_tmp += force_prefactor * rij[0];
                fy_tmp += force_prefactor * rij[1];
                fz_tmp += force_prefactor * rij[2];
            }
        }
    }, fx_i, fy_i, fz_i);

    // Update forces on atom i
    Kokkos::single(Kokkos::PerTeam(teamMember), [&]() {
        Kokkos::atomic_add(&f(i, 0), fx_i);
        Kokkos::atomic_add(&f(i, 1), fy_i);
        Kokkos::atomic_add(&f(i, 2), fz_i);
    });
}

void particles_instance::compute_ewald_reciprocal_forces() {
    int num_kpoints = (2 * k_max + 1) * (2 * k_max + 1) * (2 * k_max + 1); // Total number of k-points

    Kokkos::parallel_for("ewald-reciprocal-space-force",
        Kokkos::TeamPolicy<Tag_force_ewald_reciprocal>(num_kpoints, Kokkos::AUTO), *this);
}

KOKKOS_FUNCTION
void particles_instance::operator()(Tag_force_ewald_reciprocal, const member_type& teamMember) const {
    const int league_rank = teamMember.league_rank();
    const int num_kpoints_2D = (2 * k_max + 1) * (2 * k_max + 1);
    double alpha = ewald_alpha;

    const int kz = league_rank / num_kpoints_2D - k_max;
    const int ky = (league_rank % num_kpoints_2D) / (2 * k_max + 1) - k_max;
    const int kx = league_rank % (2 * k_max + 1) - k_max;

    if (kx == 0 && ky == 0 && kz == 0) return; // Skip the k=0 term

    double kx_real = 2 * M_PI * kx / L[0];
    double ky_real = 2 * M_PI * ky / L[1];
    double kz_real = 2 * M_PI * kz / L[2];
    double k2 = kx_real * kx_real + ky_real * ky_real + kz_real * kz_real;

    double exp_factor = exp(-k2 / (4 * alpha)) / k2;

    // Compute the structure factors S_k
    double S_re = 0.0, S_im = 0.0;

    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, N), [=](const int j, double& inner_re, double& inner_im) {
        double kr = kx_real * x(j, 0) + ky_real * x(j, 1) + kz_real * x(j, 2);
        double charge_j = charge(id(j)-1);
        inner_re += charge_j * cos(kr);
        inner_im += charge_j * sin(kr);
    }, Kokkos::Sum<double>(S_re), Kokkos::Sum<double>(S_im));

    // Compute the forces
    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, N), [=](const int i) {
        double qi = charge(id(i)-1);
        double kr_i = kx_real * x(i, 0) + ky_real * x(i, 1) + kz_real * x(i, 2);

        double sin_kr_i = sin(kr_i);
        double cos_kr_i = cos(kr_i);

        double force_prefactor = 2.0 * qi * exp_factor;

        // Force components
        double fx = force_prefactor * (S_re * sin_kr_i - S_im * cos_kr_i) * kx_real;
        double fy = force_prefactor * (S_re * sin_kr_i - S_im * cos_kr_i) * ky_real;
        double fz = force_prefactor * (S_re * sin_kr_i - S_im * cos_kr_i) * kz_real;

        // Update forces
        Kokkos::atomic_add(&f(i, 0), fx / (L[0] * L[1] * L[2]));
        Kokkos::atomic_add(&f(i, 1), fy / (L[0] * L[1] * L[2]));
        Kokkos::atomic_add(&f(i, 2), fz / (L[0] * L[1] * L[2]));
    });
}