#include "Neighbor_list.hpp"
#include <Kokkos_Core.hpp>
#include "Input_reader.hpp"
#include "particles.hpp"
#include "global.hpp"

void Neighbor_list_bonds::build_verlet_list(particles_instance& particles) {
    Kokkos::Timer Neighbor_timer;
    build_initial_verlet_list(particles);
    remove_bonds(particles);
    time_list_build += Neighbor_timer.seconds();
}

void Neighbor_list_bonds::build_initial_verlet_list(particles_instance& particles) {
    // Reset neighbor counts to zero
    Kokkos::deep_copy(this->neighbour_count, 0);
    Kokkos::deep_copy(this->verlet_list, 0);

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

                        double ry = x(i, 1) - x(j, 1);
                        ry -= int(ry * inverse_halved_L[1]) * L[1];
                        r2 += ry * ry;

                        double rz = x(i, 2) - x(j, 2);
                        rz -= int(rz * inverse_halved_L[2]) * L[2];
                        r2 += rz * rz;

                        if (r2 < (cutoff_squared * 1.2)) {
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
    // Remove bonded/angle/dihedral neighbours in one atomic-safe sweep
    Kokkos::parallel_for(
        "verlet_remove_all",
        Kokkos::TeamPolicy<Tag_verlet_remove_bonds>(N, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Tag_verlet_remove_bonds, const Kokkos::TeamPolicy<>::member_type& team) {
        const int i = team.league_rank();
        int n_initial = neighbour_count(i);
    
        // 1) Single parallel pass over the verlet slots
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, 0, n_initial),
            [&](int k) {
            int neighbour = verlet_list(i, k);
            if (neighbour == 0) return;               // already empty
    
            // check if 'neighbour' is bonded, angle- or dihedral-excluded
            bool exclude = false;
    
            // --- bonds ---
            for (size_t b = 0; b < bonds.extent(0); ++b) {
                if (bonds(b).atom1 == i && bonds(b).atom2 == neighbour ||
                    bonds(b).atom1 == neighbour && bonds(b).atom2 == i) {
                exclude = true;
                break;
                }
            }
            if (!exclude) {
                // --- angles ---
                for (size_t a = 0; a < angles.extent(0); ++a) {
                if ((angles(a).atom1 == i || angles(a).atom2 == i || angles(a).atom3 == i) &&
                    (angles(a).atom1 == neighbour || angles(a).atom2 == neighbour || angles(a).atom3 == neighbour)) {
                    exclude = true;
                    break;
                }
                }
            }
            if (!exclude) {
                // --- dihedrals ---
                for (size_t d = 0; d < dihedrals.extent(0); ++d) {
                if ((dihedrals(d).atom1 == i ||
                    dihedrals(d).atom2 == i ||
                    dihedrals(d).atom3 == i ||
                    dihedrals(d).atom4 == i) &&
                    (dihedrals(d).atom1 == neighbour ||
                    dihedrals(d).atom2 == neighbour ||
                    dihedrals(d).atom3 == neighbour ||
                    dihedrals(d).atom4 == neighbour)) {
                    exclude = true;
                    break;
                }
                }
            }
    
            if (exclude) {
                // only the first thread to swap out 'neighbour' will succeed
                int old = Kokkos::atomic_compare_exchange(
                &verlet_list(i, k),
                neighbour,      // expected
                0               // desired
                );
                if (old == neighbour) {
                Kokkos::atomic_fetch_add(&neighbour_count(i), -1);
                }
            }
            }
        );
    
        team.team_barrier();
    
        // Compact out the zeros on one thread
        Kokkos::single(Kokkos::PerTeam(team), [&]() {
            int write = 0;
            for (int read = 0; read < n_initial; ++read) {
            int val = verlet_list(i, read);
            if (val != 0) {
                verlet_list(i, write++) = val;
            }
            }
            // "write" is now the new neighbour_count
            neighbour_count(i) = write;
        });
        });

    Kokkos::fence();

    // Synchronize updated Verlet lists and neighbor counts back to the host
    Kokkos::deep_copy(h_verlet_list, verlet_list);
    Kokkos::deep_copy(h_neighbour_count, neighbour_count);

    /*for (int i = 0; i < h_verlet_list.extent(0);i++) {
        printf("%d : ",i);
        for (int j = 0; j < h_neighbour_count(i); j++) {
            printf("%d ", h_verlet_list(i,j));  
        }
        printf("\n");
    }*/
}