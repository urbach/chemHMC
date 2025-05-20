#include "global.hpp"
#include "coulomb_direct.hpp"
#include "math.h" // for definition of M_PI (value of Pi)
#include "Input_reader.hpp"

void Coulomb_direct::init(const particles_instance& particles) {

    double L[3];
    L[0] = particles.L[0];
    L[1] = particles.L[1];
    L[2] = particles.L[2];
    
    charge = Kokkos::View<double*>("charge", particles.h_atom_type_list.extent(0));
    h_charge = Kokkos::create_mirror_view(charge);

    //generate mapping from atom id to charge
    for (int i = 0; i < particles.h_atom_type_list.extent(0); i++) {
        h_charge(i) = particles.h_atom_type_list(i).charge;
    }
    Kokkos::deep_copy(charge, h_charge);
}

double Coulomb_direct::Coulomb_potential_in_cell_0(const particles_instance& particles) {
    double V = 0.0;

    // Capture all needed members of particles_instance
    auto& x = particles.x;
    auto& N = particles.N;
    auto& L = particles.L;
    auto& id = particles.id;
    auto& cutoff = this->r_c2;
    auto& charge = this->charge;
    //printf("cutoff: %f\n",cutoff);
    Kokkos::parallel_reduce(
        "coulomb-direct-potential",
        Kokkos::TeamPolicy<Tag_potential_direct>(N, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Tag_potential_direct, const Kokkos::TeamPolicy<>::member_type& teamMember, double& V) {
            const int i = teamMember.league_rank();
            double tmpV = 0.0;
            double qi = charge(id(i));

            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(teamMember, N),
                [&](const int j, double& innerV) {
                    if (i == j) return;
                    const int type_j = id(j);

                    double qj = charge(type_j);
                    double rij[3];

                    // Calculate distance vector
                    // We must not apply periodic boundaries for the direct summation
                    rij[0] = x(i, 0) - x(j, 0);
                    rij[1] = x(i, 1) - x(j, 1);
                    rij[2] = x(i, 2) - x(j, 2);

                    double r2 = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];
                    if (r2 < cutoff ) {
                        innerV += qi * qj / Kokkos::sqrt(r2);
                        //Kokkos::printf("i: %d j: %d qi: %f qj: %f rinv: %f\n",i,j,qi,qj,1/Kokkos::sqrt(r2));
                    }
                },
            tmpV); 
            
            Kokkos::single(Kokkos::PerTeam(teamMember), [&]() {
                V += tmpV;
            });
        },
    V);

    return V;
}

double Coulomb_direct::Coulomb_potential_in_cell_k(const particles_instance& particles, int k_x, int k_y, int k_z) {
    double V = 0.0;

    // Capture all needed members of particles_instance
    auto& x = particles.x;
    auto& N = particles.N;
    auto& L = particles.L;
    auto& id = particles.id;
    auto& cutoff = this->r_c2;
    auto& charge = this->charge;

    Kokkos::parallel_reduce(
        "coulomb-direct-potential",
        Kokkos::TeamPolicy<Tag_potential_direct>(N, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Tag_potential_direct, const Kokkos::TeamPolicy<>::member_type& teamMember, double& V) {
            const int i = teamMember.league_rank();
            double tmpV = 0.0;
            double qi = charge(id(i));

            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(teamMember, N),
                [&](const int j, double& innerV) {
                    const int type_j = id(j);

                    double qj = charge(type_j);
                    double rij[3];

                    // Calculate distance vector
                    // We must not apply periodic boundaries for the direct summation
                    rij[0] = x(i, 0) - (x(j, 0) + k_x * L[0]);
                    rij[1] = x(i, 1) - (x(j, 1) + k_y * L[1]);
                    rij[2] = x(i, 2) - (x(j, 2) + k_z * L[2]);

                    double r2 = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];
                    if(r2 < cutoff) {
                        innerV += qi * qj / Kokkos::sqrt(r2);
                    }
                },
            tmpV);

            Kokkos::single(Kokkos::PerTeam(teamMember), [&]() {
                V += tmpV;
            });
        },
    V);

    return V;
}

double Coulomb_direct::potential(const particles_instance& particles) {
    // Calculate potential for the simulation cell
    double V = Coulomb_potential_in_cell_0(particles);
    printf("Shell 0: %f\n",0.5 * coulombtointernal*V/kcaltointernal);
    // Calculate potential for interaction with periodic images
    const int max_shell = ceil(this->r_c/particles.L[0]);
    double shell_V;
    int k;
    // Sum over shells
    for (int shell = 1; shell <= max_shell; shell++) {
        shell_V = 0;
        for (int shell_x = -shell; shell_x <= shell; shell_x++) {
            for (int shell_y = -shell; shell_y <= shell; shell_y++) {
                for (int shell_z = -shell; shell_z <= shell; shell_z++) {
                    k = abs(shell_x) + abs(shell_y) + abs(shell_z);
                    if (k != shell) continue;
                    shell_V += Coulomb_potential_in_cell_k(particles, shell_x, shell_y, shell_z);
                }
            }
        }
        printf("Shell %d: %f\n",shell,0.5 * coulombtointernal*shell_V/kcaltointernal);
        V += shell_V;
    }
    return 0.5 * coulombtointernal * V;
}

void Coulomb_direct::force(const particles_instance& particles,type_f& f) {
    return;
    // Capture all needed members of "particles" here. We do not want to reference 
    // any members of "particle" directly inside of the kernel, as the class contains
    // functions that are not device safe. This would trigger a lot of compiler warnings.
    /*auto& x = particles.x;
    auto& N = particles.N;
    auto& L = particles.L;
    auto& id = particles.id;
    auto& verlet_list = particles.neighbor_list->verlet_list;
    auto& neighbour_count = particles.neighbor_list->neighbour_count;
    auto& inverse_L = particles.inverse_L;
    auto& charge = this->charge;
    auto& sqrt_ewald_alpha = this->sqrt_ewald_alpha;
    auto& ewald_alpha = this->ewald_alpha;
    auto& r_c2 = this->r_c2;

    typedef Kokkos::TeamPolicy<Tag_force_ewald_real> team_policy;
    Kokkos::parallel_for(
        "compute_force_ewald_real",
        team_policy(N, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Tag_force_ewald_real, const Kokkos::TeamPolicy<>::member_type& teamMember) {
            const int i = teamMember.league_rank();
            double alpha = ewald_alpha;
            double sqrt_alpha = sqrt_ewald_alpha;
            double qi = charge(id(i));

            // Temporary force accumulators for atom i
            double fx_i = 0.0;
            double fy_i = 0.0;
            double fz_i = 0.0;

            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, neighbour_count(i)), [=](const int j, double& fx_tmp, double& fy_tmp, double& fz_tmp) {
                if (i != j) {
                    const int particle_j = verlet_list(i, j);
                    const int type_j = id(particle_j);

                    double qj = charge(type_j);
                    double rij[3];

                    // Calculate distance vector and apply minimum image convention
                    rij[0] = x(i, 0) - x(particle_j, 0);
                    rij[1] = x(i, 1) - x(particle_j, 1);
                    rij[2] = x(i, 2) - x(particle_j, 2);

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
                Kokkos::atomic_add(&f(i, 0), coulombtointernal*fx_i);
                Kokkos::atomic_add(&f(i, 1), coulombtointernal*fy_i);
                Kokkos::atomic_add(&f(i, 2), coulombtointernal*fz_i);
            });
        }
    );
    Kokkos::fence();*/
}