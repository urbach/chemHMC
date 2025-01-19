#include "bonds.hpp"
#include "particles.hpp"

void Bonds::init(const particles_instance& particles) {
    int a = 0;
    return;
}

double Bonds::potential(const particles_instance& particles) {
    double result = 0.0;
    result += potential_bonds(particles);
    return 0.0;
}

void Bonds::force(const particles_instance& particles, type_f& f) {
    return;
}

double Bonds::potential_bonds(const particles_instance& particles) {
    double result = 0.0;

    // Capture all needed members of particles_instance
    auto& x = particles.x;
    auto& L = particles.L;
    auto& inverse_halved_L = particles.inverse_halved_L;
    auto& bonds = this->bonds;
    auto& bondTypes = this->bondTypes;

    // Outer parallel_reduce
    Kokkos::parallel_reduce(
        "bond-potential",
        Kokkos::TeamPolicy<Tag_potential_bonds>(bonds.extent(0), Kokkos::AUTO),
        KOKKOS_LAMBDA(const Tag_potential_bonds, const Kokkos::TeamPolicy<>::member_type& team_member, double& V) {
            const int i = team_member.league_rank();

            const int atom1 = bonds(i).atom1 - 1;
            const int atom2 = bonds(i).atom2 - 1;
            const int type = bonds(i).type - 1;
            const double k = bondTypes(type).k;
            const double r0 = bondTypes(type).r0;

            // Compute squared distance considering periodic boundaries
            double r = x(atom1, 0) - x(atom2, 0);
            r -= int(r * inverse_halved_L[0]) * L[0];
            double r2 = r * r;

            r = x(atom1, 1) - x(atom2, 1);
            r -= int(r * inverse_halved_L[1]) * L[1];
            r2 += r * r;

            r = x(atom1, 2) - x(atom2, 2);
            r -= int(r * inverse_halved_L[2]) * L[2];
            r2 += r * r;

            // Compute the bond potential
            double r_actual = sqrt(r2);
            double dr = r_actual - r0;
            double potential = k * dr * dr; // The usual factor of 0.5 is already part of k

            // Team leader updates the global potential
            Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
                V += potential;
            });
        },
    result);

    return result;
}