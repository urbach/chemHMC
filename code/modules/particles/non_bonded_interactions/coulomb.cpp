#include "global.hpp"
#include "particles.hpp"



// EWALD SUM
void particles_instance::init_ewald_sum(YAML::Node& doc) {
    ewald_alpha = check_and_assign_value<double>(doc["ewald"], "alpha");
    ewald_n_max = check_and_assign_value<int>(doc["ewald"], "N_max");
    k_max = check_and_assign_value<int>(doc["ewald"], "k_max");

    charge = Kokkos::View<double*>("charge", h_atom_type_list.extent(0));
    h_charge = Kokkos::create_mirror_view(charge);

    for (int i = 0; i < h_atom_type_list.extent(0); i++) {
        h_charge(i) = h_atom_type_list(i).charge;
    }

    Kokkos::deep_copy(charge, h_charge);
}

double particles_instance::potential_ewald_sum() {
    const double conversion_factor = 1.0;//332.062934; //conversion to kcal/mol
    double V_real = compute_ewald_real();
    double V_reciprocal = compute_ewald_reciprocal();
    double V_self = compute_ewald_self();

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
    double alpha = ewald_alpha;
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
            double r = sqrt(r2);

            innerV += qi * qj * erfc(alpha * r) / r;
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

    double exp_factor = exp(-k2 / (4 * alpha * alpha));
    tmpV += (S_re * S_re + S_im * S_im) * exp_factor / k2;

    Kokkos::single(Kokkos::PerTeam(teamMember), [&]() {
        V += tmpV;
    });
}


double particles_instance::compute_ewald_self() {
    double result = 0.0;
    Kokkos::parallel_reduce("ewald-self-energy",
        Kokkos::TeamPolicy<Tag_potential_ewald_self>(N, Kokkos::AUTO), *this, result);

    return -ewald_alpha * result / sqrt(M_PI);  // The negative sign accounts for self-interaction
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