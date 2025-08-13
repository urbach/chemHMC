#include "Neighbor_list.hpp"
#include <Kokkos_Core.hpp>
#include "Input_reader.hpp"
#include "particles.hpp"
#include "global.hpp"

void Neighbor_list::init_verlet_list(YAML::Node& doc, particles_instance& particles) {
    int N = particles.N;
    int max_neighbors;
    if (doc["particles"]["MaxNeighbors"]) {
        max_neighbors = check_and_assign_value<int>(doc["particles"], "MaxNeighbors");
    } else {
        max_neighbors = 50; // should be replaced by good estimate
    }

    // Set cutoff
    double coul_cutoff = 0.0;
    double lj_cutoff = 0.0;
    if (doc["coulomb"]["cutoff"]) {
        coul_cutoff = check_and_assign_value<double>(doc["coulomb"], "cutoff");
    }
    if (doc["LJ"]["cutoff"]) {
        lj_cutoff = check_and_assign_value<double>(doc["LJ"], "cutoff");
    }
    if (coul_cutoff > lj_cutoff) neighbor_cutoff = coul_cutoff;
    else neighbor_cutoff = lj_cutoff;

    verlet_list = Kokkos::View<int**>("verlet_list", N, max_neighbors);
    h_verlet_list = Kokkos::create_mirror_view(verlet_list);

    Kokkos::deep_copy(h_verlet_list, 0);
    Kokkos::deep_copy(verlet_list, h_verlet_list);

    neighbour_count = Kokkos::View<int*>("neighbour_count", N);
    h_neighbour_count = Kokkos::create_mirror_view(neighbour_count);
}

void Neighbor_list::build_verlet_list(particles_instance& particles) {
    Kokkos::Timer Neighbor_timer; 
    if (update_every != ++moves_since_last_update) return;
    moves_since_last_update = 0;

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

                        if (r2 >= (cutoff_squared * 1.2)) return;

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
                });
        });
    time_list_build += Neighbor_timer.seconds();
}
