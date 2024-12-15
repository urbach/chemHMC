#include "LJ.hpp"
#include "Calc.hpp"
#include "global.hpp"
#include <Kokkos_Core.hpp>
#include <functional>
#include <vector>
#include <atom.hpp>
#include <iostream>
#include <iomanip>
#include "yaml-cpp/yaml.h"
#include "particles_type.hpp"
#include "particles.hpp"
#include <cmath> // For potential calculations

void LJ::init(const particles_instance& particles) {
    std::cout << particles.N << std::endl;
    h_epsilon_mat = particles.h_epsilon_mat;
    h_sigma_mat = particles.h_sigma_mat;
    // Allocate and copy the device views for epsilon_mat and sigma_mat
    epsilon_mat = Kokkos::View<double**>("epsilon_mat", h_epsilon_mat.extent(0), h_epsilon_mat.extent(1));
    sigma_mat = Kokkos::View<double**>("sigma_mat", h_sigma_mat.extent(0), h_sigma_mat.extent(1));
    // Copy data from host to device memory so its accessible in the kernel
    Kokkos::deep_copy(epsilon_mat, h_epsilon_mat); 
    Kokkos::deep_copy(sigma_mat, h_sigma_mat);
    L[0] = particles.L[0];
    L[1] = particles.L[1];
    L[2] = particles.L[2];
    inverse_L[0] = particles.inverse_L[0];
    inverse_L[1] = particles.inverse_L[1];
    inverse_L[2] = particles.inverse_L[2];
    inverse_halved_L[0] = particles.inverse_halved_L[0];
    inverse_halved_L[1] = particles.inverse_halved_L[1];
    inverse_halved_L[2] = particles.inverse_halved_L[2];
    cutoff_squared = particles.cutoff_squared;
}

double LJ::potential(const particles_instance& particles) {
    double result;
    auto& x = particles.x;
    auto& id = particles.id;

    // Capture all necessary variables explicitly
    auto& L = this->L;
    auto& inverse_halved_L = this->inverse_halved_L;
    auto& cutoff_squared = this->cutoff_squared;
    auto& sigma_mat = this->sigma_mat;
    auto& epsilon_mat = this->epsilon_mat;

    // Outer parallel_reduce
    Kokkos::parallel_reduce(
        "particles-LJ-potential-AMICAIP",
        Kokkos::TeamPolicy<Tag_potential_AMIC_inner_parallel>(x.extent(0), Kokkos::AUTO),
        KOKKOS_LAMBDA(const Tag_potential_AMIC_inner_parallel, const member_type& teamMember, double& V) {
            const int i = teamMember.league_rank();
            double tmpV = 0.0;
            const int type_i = id(i) - 1;

            // Perform inner reduction outside nested lambda
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(teamMember, i + 1, x.extent(0)),
                [&](const int j, double& innerV) {
                    if (i != j) {
                        const int type_j = id(j) - 1;

                        double rij = x(i, 0) - x(j, 0);
                        rij -= int(rij * inverse_halved_L[0]) * L[0];
                        double r2 = rij * rij;
                        if (r2 > cutoff_squared) return;

                        rij = x(i, 1) - x(j, 1);
                        rij -= int(rij * inverse_halved_L[1]) * L[1];
                        r2 += rij * rij;
                        if (r2 > cutoff_squared) return;

                        rij = x(i, 2) - x(j, 2);
                        rij -= int(rij * inverse_halved_L[2]) * L[2];
                        r2 += rij * rij;

                        if (r2 < cutoff_squared) {
                            double sigma = sigma_mat(type_i, type_j);
                            double epsilon = epsilon_mat(type_i, type_j);
                            double sr2 = sigma * sigma / r2;
                            double sr6 = sr2 * sr2 * sr2;
                            innerV += epsilon * sr6 * (sr6 - 1.0);
                        }
                    }
                },
                tmpV);

            // Team leader updates the global potential V
            Kokkos::single(Kokkos::PerTeam(teamMember), [&]() {
                V += tmpV;
            });
        },
        result);

    return 4.0 * result;
}

void LJ::force(const particles_instance& particles, type_f& f) {
    // Set refs to needed members of particle here. We do not want to reference 
    // any members of particle directly inside of the kernel, as the class contains
    // functions that are not device safe. This will trigger a lot of compiler warnings.
    auto& x = particles.x;
    auto& id = particles.id;

    // explicitly capture all needed members for the kernel
    auto& L = this->L;
    auto& inverse_halved_L = this->inverse_halved_L;
    auto& cutoff_squared = this->cutoff_squared;
    auto& sigma_mat = this->sigma_mat;
    auto& epsilon_mat = this->epsilon_mat;

    typedef Kokkos::TeamPolicy<Tag_force_AMIC_inner_parallel> team_policy;
    // Launch the parallel kernel
    Kokkos::parallel_for("particles-LJ-force-AMICAIP", team_policy(particles.N, Kokkos::AUTO),
                         KOKKOS_LAMBDA(const Tag_force_AMIC_inner_parallel, const member_type& teamMember) {
        const int i = teamMember.league_rank();
        const int type_i = id(i) - 1;

        Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, i + 1, x.extent(0)),
            [=](const int j) {
                if (i != j) {
                    const int type_j = id(j) - 1;

                    // Calculate minimum image distance
                    double rx = x(i, 0) - x(j, 0);
                    rx -= int(rx * inverse_halved_L[0]) * L[0];
                    double r2 = rx * rx;
                    if (r2 > cutoff_squared) return;

                    double ry = x(i, 1) - x(j, 1);
                    ry -= int(ry * inverse_halved_L[1]) * L[1];
                    r2 += ry * ry;
                    if (r2 > cutoff_squared) return;

                    double rz = x(i, 2) - x(j, 2);
                    rz -= int(rz * inverse_halved_L[2]) * L[2];
                    r2 += rz * rz;

                    // Apply Lennard-Jones force calculations
                    if (r2 < cutoff_squared) {
                        double sr2 = sigma_mat(type_i, type_j) * sigma_mat(type_i, type_j) / r2;
                        double sr6 = sr2 * sr2 * sr2;
                        sr2 = sr6 * (-sr6 + 0.5) / r2;
                        double force = 48.0 * epsilon_mat(type_i, type_j) * sr2;

                        // Accumulate forces for particle i
                        Kokkos::atomic_add(&f(i, 0), force * rx);
                        Kokkos::atomic_add(&f(i, 1), force * ry);
                        Kokkos::atomic_add(&f(i, 2), force * rz);

                        Kokkos::atomic_add(&f(j, 0), -force * rx);
                        Kokkos::atomic_add(&f(j, 1), -force * ry);
                        Kokkos::atomic_add(&f(j, 2), -force * rz);
                    }
                }
            });
    });
    Kokkos::fence();
}