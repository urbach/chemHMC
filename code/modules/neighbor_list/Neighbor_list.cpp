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
    // square the cutoff and add a skin distance
    double skin_distance = neighbor_cutoff * 0.1;
    neighbor_cutoff_squared = (neighbor_cutoff + skin_distance) * (neighbor_cutoff + skin_distance);
    skin_distance_squared = skin_distance * skin_distance;
    verlet_list = Kokkos::View<int**>("verlet_list", N, max_neighbors);
    h_verlet_list = Kokkos::create_mirror_view(verlet_list);

    x_last = Kokkos::View<double*[3]>("x_last", N);
    h_x_last = Kokkos::create_mirror_view(x_last);

    disp2 = Kokkos::View<double*>("disp2", N);
    h_disp2 = Kokkos::create_mirror_view(disp2);

    Kokkos::deep_copy(h_verlet_list, 0);
    Kokkos::deep_copy(verlet_list, h_verlet_list);

    neighbour_count = Kokkos::View<int*>("neighbour_count", N);
    h_neighbour_count = Kokkos::create_mirror_view(neighbour_count);
}

void Neighbor_list::build_verlet_list(particles_instance& particles) {
    Kokkos::Timer neighbor_timer;
    // First check if atoms have moved sufficiently since last build, if not,
    // we can skip a new build
    const int N  = particles.N;
    const double L0 = particles.L[0], L1 = particles.L[1], L2 = particles.L[2];
    const double ihL0 = particles.inverse_halved_L[0];
    const double ihL1 = particles.inverse_halved_L[1];
    const double ihL2 = particles.inverse_halved_L[2];
    auto x     = particles.x;
    auto xlast = x_last;
    auto disp  = disp2;

    double max_disp2 = 0.0;
    Kokkos::parallel_reduce(
        "update_and_max_disp2",
        Kokkos::RangePolicy(0, N),
        KOKKOS_LAMBDA (const int i, double& lmax) {
            double dx = x(i,0) - xlast(i,0);
            dx -= int(dx * ihL0) * L0;

            double dy = x(i,1) - xlast(i,1);
            dy -= int(dy * ihL1) * L1;

            double dz = x(i,2) - xlast(i,2);
            dz -= int(dz * ihL2) * L2;

            double d2 = dx*dx + dy*dy + dz*dz;
            disp(i) += d2;
            lmax = (lmax < disp(i)) ? disp(i) : lmax;
        },
        Kokkos::Max<double>(max_disp2)
    );

    if (max_disp2 >= skin_distance_squared) {
        build(particles);
        Kokkos::deep_copy(x_last, x);
        Kokkos::deep_copy(disp2, 0.0);
    }

    time_list_build += neighbor_timer.seconds();
}

void Neighbor_list::build(particles_instance& particles) {
    // Reset neighbor counts to zero
    Kokkos::deep_copy(this->neighbour_count, 0);
    Kokkos::deep_copy(this->verlet_list, 0);

    // Capture all needed members of particles_instance
    auto& x = particles.x;
    auto& neighbour_count = this->neighbour_count;
    auto& verlet_list = this->verlet_list;
    auto& L = particles.L;
    auto& N = particles.N;
    auto& inverse_halved_L = particles.inverse_halved_L;
    auto& cutoff_squared = this->neighbor_cutoff_squared;

    const double ihL0 = inverse_halved_L[0];
    const double ihL1 = inverse_halved_L[1];
    const double ihL2 = inverse_halved_L[2];
    const double L0 = L[0], L1 = L[1], L2 = L[2];
    const int neighbor_count_length = neighbour_count.extent(0);

    Kokkos::parallel_for(
        "populate_verlet_list",
        Kokkos::TeamPolicy<Tag_build_verlet_list>(N, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Tag_build_verlet_list, const Kokkos::TeamPolicy<>::member_type& teamMember) {
            const int i = teamMember.league_rank();
            const double xi0 = x(i,0), xi1 = x(i,1), xi2 = x(i,2);

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(teamMember, i+1, N),
                [&](const int j) {
                    double r_tmp = xi0 - x(j, 0);
                    r_tmp -= int(r_tmp * ihL0) * L0;
                    double r2 = r_tmp * r_tmp;

                    r_tmp = xi1 - x(j, 1);
                    r_tmp -= int(r_tmp * ihL1) * L1;
                    r2 += r_tmp * r_tmp;

                    r_tmp = xi2 - x(j, 2);
                    r_tmp -= int(r_tmp * ihL2) * L2;
                    r2 += r_tmp * r_tmp;

                    if (r2 >= (cutoff_squared)) return;

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
                });
        });
    // Copy to x_last so we can compare future steps to this build
    Kokkos::deep_copy(x_last, particles.x);
}
