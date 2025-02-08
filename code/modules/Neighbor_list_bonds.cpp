#include "Neighbor_list.hpp"
#include <Kokkos_Core.hpp>
#include "Input_reader.hpp"
#include "particles.hpp"
#include "global.hpp"

void Neighbor_list_bonds::build_verlet_list(particles_instance& particles) {
    build_initial_verlet_list(particles);
    remove_bonds(particles);
}

void Neighbor_list_bonds::build_initial_verlet_list(particles_instance& particles) {
    // Reset neighbor counts to zero
    Kokkos::deep_copy(this->neighbour_count, 0);

    // Capture all needed members of particles_instance
    auto& x = particles.x;
    auto& neighbour_count = this->neighbour_count;
    auto& verlet_list = this->verlet_list;
    auto& L = particles.L;
    auto& inverse_halved_L = particles.inverse_halved_L;
    auto& cutoff_squared = particles.cutoff_squared;
    // Outer parallel_for
    Kokkos::parallel_for(
        "populate_verlet_list",
        Kokkos::TeamPolicy<Tag_build_verlet_list>(x.extent(0), Kokkos::AUTO),
        KOKKOS_LAMBDA(const Tag_build_verlet_list, const Kokkos::TeamPolicy<>::member_type& teamMember) {
            const int i = teamMember.league_rank();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(teamMember, neighbour_count.extent(0)),
                [&](const int j) {
                    if (i < j) {
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

                        if (r2 < (cutoff_squared * 1.3)) {
                            int current_count = Kokkos::atomic_fetch_add(&neighbour_count(i), 1);
                            if (current_count < verlet_list.extent(1)) {
                                verlet_list(i, current_count) = j;
                            } else {
                                Kokkos::printf("COUNT: %d\n", current_count);
                                Kokkos::abort(
                                    "ERROR: Verlet list dimension not large enough for the number of neighbors! "
                                    "Increase it manually in the config file with MaxParticles."
                                );
                            }
                        }
                    }
                });
        });

    Kokkos::deep_copy(h_neighbour_count,neighbour_count);
    Kokkos::deep_copy(h_verlet_list,verlet_list);
}

void Neighbor_list_bonds::remove_bonds(particles_instance& particles) {
    // Remove bonded and indirectly bonded atoms (angles, dihedrals) from the neighbor list
    
    // Capture required variables explicitly
    auto& N = particles.N;
    auto& neighbour_count = this->neighbour_count;
    auto& verlet_list = this->verlet_list;
    auto& bonds = particles.bonds_ptr->bonds;
    auto& angles = particles.bonds_ptr->angles;
    auto& dihedrals = particles.bonds_ptr->dihedrals;

    // Remove bonded atoms and associated interactions
    Kokkos::parallel_for(
        "verlet_remove_bonds",
        Kokkos::TeamPolicy<Tag_verlet_remove_bonds>(N, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Tag_verlet_remove_bonds, const Kokkos::TeamPolicy<>::member_type& teamMember) {
            const int i = teamMember.league_rank();

            int initial_neighbour_count = neighbour_count(i);

            // Remove bonded atoms
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(teamMember, bonds.extent(0)),
                [&](const int b) {
                    int atom1 = bonds(b).atom1;
                    int atom2 = bonds(b).atom2;
                    if (atom1 != i) return;

                    // Search for atom2 in the Verlet list of atom i and mark for removal
                    int n = initial_neighbour_count;
                    for (int k = 0; k < n; ++k) {
                        if (verlet_list(i, k) == atom2) {
                            verlet_list(i, k) = 0; // Mark for removal
                            Kokkos::atomic_fetch_add(&neighbour_count(i), -1);
                            break;
                        }
                    }
                });

            // Remove atoms indirectly bonded via angles
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(teamMember, angles.extent(0)),
                [&](const int a) {
                    int atom1 = angles(a).atom1;
                    int atom2 = angles(a).atom2;
                    int atom3 = angles(a).atom3;

                    if (atom1 != i) return;

                    // Remove atom2 and atom3 from Verlet list
                    int n = initial_neighbour_count;
                    for (int k = 0; k < n; ++k) {
                        if (verlet_list(i, k) == atom2 || verlet_list(i, k) == atom3) {
                            verlet_list(i, k) = 0; // Mark for removal
                            Kokkos::atomic_fetch_add(&neighbour_count(i), -1);
                        }
                    }
                });

            // Remove atoms connected via dihedrals
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(teamMember, dihedrals.extent(0)),
                [&](const int d) {
                    int atom1 = dihedrals(d).atom1;
                    int atom2 = dihedrals(d).atom2;
                    int atom3 = dihedrals(d).atom3;
                    int atom4 = dihedrals(d).atom4;

                    if (atom1 != i) return;

                    // Remove atom2, atom3, and atom4 from Verlet list
                    int n = initial_neighbour_count;
                    for (int k = 0; k < n; ++k) {
                        if (verlet_list(i, k) == atom2 || verlet_list(i, k) == atom3 || verlet_list(i, k) == atom4) {
                            verlet_list(i, k) = 0; // Mark for removal
                            Kokkos::atomic_fetch_add(&neighbour_count(i), -1);
                        }
                    }
                }
            );
            teamMember.team_barrier();
            // Clean up list to remove zeros and shift valid entries up
            Kokkos::single(Kokkos::PerTeam(teamMember), [&]() {
                int n = initial_neighbour_count;
                int write_idx = 0;

                for (int read_idx = 0; read_idx < n; ++read_idx) {
                    if (verlet_list(i, read_idx) != 0) {
                        verlet_list(i, write_idx) = verlet_list(i, read_idx);
                        ++write_idx;
                    }
                }
            });
        });

    Kokkos::fence();

    // Synchronize updated Verlet lists and neighbor counts back to the host
    Kokkos::deep_copy(h_verlet_list, verlet_list);
    Kokkos::deep_copy(h_neighbour_count, neighbour_count);
}