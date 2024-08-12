#include "global.hpp"
#include "particles.hpp"

double particles_instance::potential_all_neighbour_inner_parallel() {
    double result;
    Kokkos::parallel_reduce("particles-LJ-potential-all-inner-parallel",
        Kokkos::TeamPolicy<Tag_potential_all_inner_parallel>(N, Kokkos::AUTO), *this, result);
    // 2 *eps instead of 4 *eps because we count the couples i,j twice
    return 2 * result;
}

KOKKOS_FUNCTION
void particles_instance::operator() (Tag_potential_all_inner_parallel, const member_type& teamMember, double& V) const {
    const int i = teamMember.league_rank();
    double tmpV = 0;
    int type_i = id[i]-1;
    
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, N), [=](const int j, double& innerV) {
        for (int bx = -1; bx < 2; bx++) {
            for (int by = -1; by < 2; by++) {
                for (int bz = -1; bz < 2; bz++) {
                    if (!(i == j && bx == 0 && by == 0 && bz == 0)) {
                        int type_j = id[j]-1;
                        double  rij = x(i, 0) - (x(j, 0) + bx * L[0]);
                        double  r2 = rij * rij;
                        rij = x(i, 1) - (x(j, 1) + by * L[1]);
                        r2 += rij * rij;
                        rij = x(i, 2) - (x(j, 2) + bz * L[2]);
                        r2 += rij * rij;


                        if (r2 < cutoff_squared) {
                            double sr2 = sigma_mat(type_i,type_j) * sigma_mat(type_i,type_j) / r2;
                            double sr6 = sr2 * sr2 * sr2;
                            innerV += epsilon_mat(type_i,type_j) * sr6 * (sr6 - 1.0);
                        }
                    }
                }
            }
        }
    }, tmpV);
    Kokkos::single(Kokkos::PerTeam(teamMember), [&]() {
        V += tmpV;
        });
}

void particles_instance::compute_force_all_inner_parallel() {
    typedef Kokkos::TeamPolicy<Tag_force_inner_parallel>  team_policy;
    Kokkos::parallel_for("particles-LJ-force-all-inner-parallel", team_policy(N, Kokkos::AUTO), *this);
}

KOKKOS_FUNCTION
void particles_instance::operator() (Tag_force_inner_parallel, const member_type& teamMember) const {
    const int i = teamMember.league_rank();// bin id
    f(i, 0) = 0;
    f(i, 1) = 0;
    f(i, 2) = 0;
    int type_i = id[i]-1;
    space_vector  fv;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, N), [=](const int j, space_vector& innerfv) {
        for (int bx = -1; bx < 2; bx++) {
            for (int by = -1; by < 2; by++) {
                for (int bz = -1; bz < 2; bz++) {
                    if (!(i == j && bx == 0 && by == 0 && bz == 0)) {
                        int type_j = id[j]-1;
                        double  rij = x(i, 0) - (x(j, 0) + bx * L[0]);
                        double r2 = rij * rij;
                        rij = x(i, 1) - (x(j, 1) + by * L[1]);
                        r2 += rij * rij;
                        rij = x(i, 2) - (x(j, 2) + bz * L[2]);
                        r2 += rij * rij;

                        if (r2 < cutoff_squared) {
                            double sr2 = sigma_mat(type_i,type_j) * sigma_mat(type_i,type_j) / r2;
                            double sr6 = sr2 * sr2 * sr2;
                            sr2 = sr6 * (-sr6 + 0.5) / r2;
                            innerfv.the_array[0] += epsilon_mat(type_i,type_j) * sr2 * (x(i, 0) - (x(j, 0) + bx * L[0]));
                            innerfv.the_array[1] += epsilon_mat(type_i,type_j) * sr2 * (x(i, 1) - (x(j, 1) + by * L[1]));
                            innerfv.the_array[2] += epsilon_mat(type_i,type_j) * sr2 * (x(i, 2) - (x(j, 2) + bz * L[2]));
                        }
                    }
                }
            }
        }
        }, fv);
    f(i, 0) = fv.the_array[0] * 48;
    f(i, 1) = fv.the_array[1] * 48;
    f(i, 2) = fv.the_array[2] * 48;

}

double particles_instance::potential_MICAIP() {
    double result;
    Kokkos::parallel_reduce("particles-LJ-potential-MICAIP",
        Kokkos::TeamPolicy<Tag_potential_MIC_inner_parallel>(N, Kokkos::AUTO), *this, result);
    // 2 *eps instead of 4 *eps because we count the couples i,j twice
    return 2 * result;
}

KOKKOS_FUNCTION
void particles_instance::operator() (Tag_potential_MIC_inner_parallel, const member_type& teamMember, double& V) const {
    const int i = teamMember.league_rank();
    double tmpV = 0;
    int type_i = id[i]-1;
    
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, N), [=](const int j, double& innerV) {
        if (!(i == j)) {
            int type_j = id[j]-1;
            double rij = x(i, 0) - x(j, 0);
            rij -= int(rij*inverse_halved_L[0]) * L[0];
            double  r2 = rij * rij;
            rij = x(i, 1) - x(j, 1);
            rij -= int(rij*inverse_halved_L[1]) * L[1];
            r2 += rij * rij;
            rij = x(i, 2) - x(j, 2);
            rij -= int(rij*inverse_halved_L[2]) * L[2];
            r2 += rij * rij;

            if (r2 < cutoff_squared) {
                double sr2 = sigma_mat(type_i,type_j) * sigma_mat(type_i,type_j) / r2;
                double sr6 = sr2 * sr2 * sr2;
                innerV += epsilon_mat(type_i,type_j) * sr6 * (sr6 - 1.0);
            }
        }
    }, tmpV);
    Kokkos::single(Kokkos::PerTeam(teamMember), [&]() {
        V += tmpV;
        });
}

void particles_instance::compute_force_MICAIP() {
    typedef Kokkos::TeamPolicy<Tag_force_MIC_inner_parallel>  team_policy;
    Kokkos::parallel_for("particles-LJ-force-MICAIP", team_policy(N, Kokkos::AUTO), *this);
}

double particles_instance::potential_AMICAIP() {
    double result;
    Kokkos::parallel_reduce("particles-LJ-potential-AMICAIP",
        Kokkos::TeamPolicy<Tag_potential_AMIC_inner_parallel>(N, Kokkos::AUTO), *this, result);
    return 4 * result;
}

KOKKOS_FUNCTION
void particles_instance::operator() (Tag_potential_AMIC_inner_parallel, const member_type& teamMember, double& V) const {
    const int i = teamMember.league_rank();
    double tmpV = 0;
    int type_i = id[i]-1;
    
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, i+1,N), [=](const int j, double& innerV) {
        if (!(i == j)) {
            int type_j = id[j]-1;
            double rij = x(i, 0) - x(j, 0);
            rij -= int(rij*inverse_halved_L[0]) * L[0];
            double  r2 = rij * rij;
            rij = x(i, 1) - x(j, 1);
            rij -= int(rij*inverse_halved_L[1]) * L[1];
            r2 += rij * rij;
            rij = x(i, 2) - x(j, 2);
            rij -= int(rij*inverse_halved_L[2]) * L[2];
            r2 += rij * rij;

            if (r2 < cutoff_squared) {
                double sr2 = sigma_mat(type_i,type_j) * sigma_mat(type_i,type_j) / r2;
                double sr6 = sr2 * sr2 * sr2;
                innerV += epsilon_mat(type_i,type_j) * sr6 * (sr6 - 1.0);
            }
        }
    }, tmpV);
    Kokkos::single(Kokkos::PerTeam(teamMember), [&]() {
        V += tmpV;
        });
}

KOKKOS_FUNCTION
void particles_instance::operator() (Tag_force_MIC_inner_parallel, const member_type& teamMember) const {
    const int i = teamMember.league_rank();// bin id
    f(i, 0) = 0;
    f(i, 1) = 0;
    f(i, 2) = 0;
    int type_i = id[i]-1;
    space_vector  fv;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, N), [=](const int j, space_vector& innerfv) {
        if (!(i == j)) {
            int type_j = id[j]-1;
            // calculate minimum image distance in each direction
            double rx = x(i, 0) - x(j, 0);
            rx -= int(rx*inverse_halved_L[0]) * L[0];
            double r2 = rx*rx;
            double ry = x(i, 1) - x(j, 1);
            ry -= int(ry*inverse_halved_L[1]) * L[1];
            r2 += ry * ry;
            double rz = x(i, 2) - x(j, 2);
            rz -= int(rz*inverse_halved_L[2]) * L[2];
            r2 += rz * rz;


            if (r2 < cutoff_squared) {
                double sr2 = sigma_mat(type_i,type_j) * sigma_mat(type_i,type_j) / r2;
                double sr6 = sr2 * sr2 * sr2;
                sr2 = sr6 * (-sr6 + 0.5) / r2;
                innerfv.the_array[0] += epsilon_mat(type_i,type_j) * sr2 * rx;
                innerfv.the_array[1] += epsilon_mat(type_i,type_j) * sr2 * ry;
                innerfv.the_array[2] += epsilon_mat(type_i,type_j) * sr2 * rz;
            }
        }
    }, fv);
    f(i, 0) = fv.the_array[0] * 48;
    f(i, 1) = fv.the_array[1] * 48;
    f(i, 2) = fv.the_array[2] * 48;

}

void particles_instance::compute_force_AMICAIP() {
    typedef Kokkos::TeamPolicy<Tag_force_AMIC_inner_parallel>  team_policy;
    Kokkos::parallel_for("particles-LJ-force-AMICAIP", team_policy(N, Kokkos::AUTO), *this);
}

KOKKOS_FUNCTION
void particles_instance::operator() (Tag_force_AMIC_inner_parallel, const member_type& teamMember) const {
    const int i = teamMember.league_rank();// bin id
    f(i, 0) = 0;
    f(i, 1) = 0;
    f(i, 2) = 0;
    int type_i = id[i]-1;
    space_vector  fv;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, i+1 ,N), [=](const int j, space_vector& innerfv) {
        if (!(i == j)) {
            int type_j = id[j]-1;
            // calculate minimum image distance
            double rx = x(i, 0) - x(j, 0);
            rx -= int(rx*inverse_halved_L[0]) * L[0];
            double r2 = rx*rx;
            double ry = x(i, 1) - x(j, 1);
            ry -= int(ry*inverse_halved_L[1]) * L[1];
            r2 += ry * ry;
            double rz = x(i, 2) - x(j, 2);
            rz -= int(rz*inverse_halved_L[2]) * L[2];
            r2 += rz * rz;


            if (r2 < cutoff_squared) {
            double sr2 = sigma_mat(type_i, type_j) * sigma_mat(type_i, type_j) / r2;
            double sr6 = sr2 * sr2 * sr2;
            sr2 = sr6 * (-sr6 + 0.5) / r2;
            double force = 48 * epsilon_mat(type_i, type_j) * sr2;

            Kokkos::atomic_add(&f(i, 0), force * rx);
            Kokkos::atomic_add(&f(i, 1), force * ry);
            Kokkos::atomic_add(&f(i, 2), force * rz);

            Kokkos::atomic_add(&f(j, 0), -force * rx);
            Kokkos::atomic_add(&f(j, 1), -force * ry);
            Kokkos::atomic_add(&f(j, 2), -force * rz);
            }
        }
    }, fv);
}