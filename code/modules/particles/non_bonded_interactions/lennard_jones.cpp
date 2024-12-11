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
    Kokkos::parallel_reduce("LJ-potential-MICAIP",
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
            if (r2 > cutoff_squared) return;
            rij = x(i, 1) - x(j, 1);
            rij -= int(rij*inverse_halved_L[1]) * L[1];
            r2 += rij * rij;
            if (r2 > cutoff_squared) return;
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
        //Kokkos::printf("Accumulated tmpV = %f\n", tmpV);
        });
    //Kokkos::printf("Accumulated potential for particle %d = %f\n", i, tmpV);
}

void particles_instance::compute_force_AMICAIP() {
    // Set all forces to 0
    Kokkos::deep_copy(f,0);
    typedef Kokkos::TeamPolicy<Tag_force_AMIC_inner_parallel>  team_policy;
    Kokkos::parallel_for("particles-LJ-force-AMICAIP", team_policy(N, Kokkos::AUTO), *this);
}

KOKKOS_FUNCTION
void particles_instance::operator() (Tag_force_AMIC_inner_parallel, const member_type& teamMember) const {
    const int i = teamMember.league_rank();
    int type_i = id[i]-1;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, i+1 ,N), [=](const int j) {
        if (!(i == j)) {
            int type_j = id[j]-1;
            // calculate minimum image distance
            double rx = x(i, 0) - x(j, 0);
            rx -= int(rx*inverse_halved_L[0]) * L[0];
            double r2 = rx*rx;
            if (r2 > cutoff_squared) return;
            double ry = x(i, 1) - x(j, 1);
            ry -= int(ry*inverse_halved_L[1]) * L[1];
            r2 += ry * ry;
            if (r2 > cutoff_squared) return;
            double rz = x(i, 2) - x(j, 2);
            rz -= int(rz*inverse_halved_L[2]) * L[2];
            r2 += rz * rz;


            if (r2 < cutoff_squared) {
            double sr2 = sigma_mat(type_i, type_j) * sigma_mat(type_i, type_j) / r2;
            double sr6 = sr2 * sr2 * sr2;
            sr2 = sr6 * (-sr6 + 0.5) / r2;
            double force = 48 * epsilon_mat(type_i, type_j) * sr2;

            // Use atomic add to avoid race conditions
            Kokkos::atomic_add(&f(i, 0), force * rx);
            Kokkos::atomic_add(&f(i, 1), force * ry);
            Kokkos::atomic_add(&f(i, 2), force * rz);

            Kokkos::atomic_add(&f(j, 0), -force * rx);
            Kokkos::atomic_add(&f(j, 1), -force * ry);
            Kokkos::atomic_add(&f(j, 2), -force * rz);
            }
        }
    });
}

double particles_instance::potential_cell_list() {
    double result = 0.0;
    // rebuild cell list after every step
    populate_cell_list();
    Kokkos::parallel_reduce("particles-LJ-potential-cell-list",
        Kokkos::TeamPolicy<Tag_potential_cell>(h_cells_per_dim(0) * h_cells_per_dim(1) * h_cells_per_dim(2), Kokkos::AUTO), *this, result);
    return 4 * result;
}

KOKKOS_FUNCTION
void particles_instance::operator()(Tag_potential_cell, const member_type& teamMember, double& V) const {
    const int cell_index = teamMember.league_rank();  // Index of the current cell
    double tmpV = 0.0;
    // Get the number of particles in current cell
    int num_particles_in_cell = cell_count(cell_index);
    
    // Loop over particles in the current cell
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, num_particles_in_cell), 
    [=](int i, double& tmp_innerV) {
        int particle_i = cell_list(cell_index, i);
        int type_i = id(particle_i) - 1;

        // Loop over neighboring cells
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int neighbor_cell_index = compute_neighbor_cell_index(cell_index, dx, dy, dz);
                    if (neighbor_cell_index >= 0 && neighbor_cell_index < cells_per_dim[0] * cells_per_dim[1] * cells_per_dim[2]) {
                        int num_particles_in_neighbor_cell = cell_count(neighbor_cell_index);
                        double innerV = 0.0;
                        // Loop over particles in currently selected neighbour cell
                        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(teamMember, num_particles_in_neighbor_cell),
                        [=, &innerV](int j, double& local_innerV) {
                            int particle_j = cell_list(neighbor_cell_index, j);
                            
                            // Avoid double counting of potential
                            if (particle_i < particle_j) {
                                int type_j = id(particle_j) - 1;

                                double rx = x(particle_i, 0) - x(particle_j, 0);
                                rx -= int(rx * inverse_halved_L[0]) * L[0];
                                double r2 = rx * rx;
                                if (r2 > cutoff_squared) return;
                                double ry = x(particle_i, 1) - x(particle_j, 1);
                                ry -= int(ry * inverse_halved_L[1]) * L[1];
                                r2 += ry * ry;
                                if (r2 > cutoff_squared) return;
                                double rz = x(particle_i, 2) - x(particle_j, 2);
                                rz -= int(rz * inverse_halved_L[2]) * L[2];
                                r2 += rz * rz;

                                if (r2 < cutoff_squared) {
                                    
                                    double sr2 = sigma_mat(type_i, type_j) * sigma_mat(type_i, type_j) / r2;
                                    double sr6 = sr2 * sr2 * sr2;
                                    local_innerV += epsilon_mat(type_i, type_j) * sr6 * (sr6 - 1.0);
                                    // Kokkos::printf("i: %d j: %d pot: %f\n", particle_i, particle_j, local_innerV);
                                }
                                
                            }
                        }, innerV);
                        tmp_innerV += innerV;
                    }
                }
            }
        }
        //Kokkos::printf("Accumulated potential for particle %d = %f\n", particle_i, tmp_innerV);
    }, tmpV);

    Kokkos::single(Kokkos::PerTeam(teamMember), [&]() {
        V += tmpV;
        //Kokkos::printf("Accumulated tmpV = %f\n", tmpV);
    });
}

void particles_instance::compute_force_cell_list() {
    // Set all forces to 0
    Kokkos::deep_copy(f,0);
    populate_cell_list();
    typedef Kokkos::TeamPolicy<Tag_force_cell> team_policy;
    Kokkos::parallel_for("compute_force_cell_list", team_policy(h_cells_per_dim(0) * h_cells_per_dim(1) * h_cells_per_dim(2), Kokkos::AUTO), *this);
    Kokkos::fence();
}

KOKKOS_FUNCTION
void particles_instance::operator()(Tag_force_cell, const member_type& teamMember) const {
    const int cell_index = teamMember.league_rank();  // Index of the current cell
    // Get the number of particles in current cell
    int num_particles_in_cell = cell_count(cell_index);
    
    // Loop over particles in the current cell
    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, num_particles_in_cell), 
    [=](int i) {
        int particle_i = cell_list(cell_index, i);
        int type_i = id(particle_i) - 1;

        // Loop over neighboring cells
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int neighbor_cell_index = compute_neighbor_cell_index(cell_index, dx, dy, dz);
                    if (neighbor_cell_index >= 0 && neighbor_cell_index < cells_per_dim[0] * cells_per_dim[1] * cells_per_dim[2]) {
                        int num_particles_in_neighbor_cell = cell_count(neighbor_cell_index);

                        // Loop over particles in currently selected neighbour cell
                        Kokkos::parallel_for(Kokkos::ThreadVectorRange(teamMember, num_particles_in_neighbor_cell),
                        [=](int j) {
                            int particle_j = cell_list(neighbor_cell_index, j);
                            
                            // Avoid double counting of force
                            if (particle_i < particle_j) {
                                int type_j = id(particle_j) - 1;

                                double rx = x(particle_i, 0) - x(particle_j, 0);
                                rx -= int(rx * inverse_halved_L[0]) * L[0];
                                double r2 = rx * rx;
                                if (r2 > cutoff_squared) return;
                                double ry = x(particle_i, 1) - x(particle_j, 1);
                                ry -= int(ry * inverse_halved_L[1]) * L[1];
                                r2 += ry * ry;
                                if (r2 > cutoff_squared) return;
                                double rz = x(particle_i, 2) - x(particle_j, 2);
                                rz -= int(rz * inverse_halved_L[2]) * L[2];
                                r2 += rz * rz;

                                if (r2 < cutoff_squared) {
                                    double sr2 = sigma_mat(type_i, type_j) * sigma_mat(type_i, type_j) / r2;
                                    double sr6 = sr2 * sr2 * sr2;
                                    sr2 = sr6 * (-sr6 + 0.5) / r2;
                                    double force = 48 * epsilon_mat(type_i, type_j) * sr2;
                                    //Kokkos::printf("force: %f \n", force);
                                    // Use atomic add to avoid race conditions
                                    Kokkos::atomic_add(&f(particle_i, 0), force * rx);
                                    Kokkos::atomic_add(&f(particle_i, 1), force * ry);
                                    Kokkos::atomic_add(&f(particle_i, 2), force * rz);

                                    Kokkos::atomic_add(&f(particle_j, 0), -force * rx);
                                    Kokkos::atomic_add(&f(particle_j, 1), -force * ry);
                                    Kokkos::atomic_add(&f(particle_j, 2), -force * rz);
                                }
                            }
                        });
                    }
                }
            }
        }
    });
} 

void particles_instance::populate_cell_list() {
    // Reset the cell counts to zero
    Kokkos::deep_copy(cell_count, 0);
    // Populate the cell list
    typedef Kokkos::RangePolicy<Tag_populate_cell_list> range_policy;
    Kokkos::parallel_for("populate_cell_list", range_policy(0, N), *this);
    Kokkos::fence();
    Kokkos::deep_copy(h_cell_list, cell_list);
}

KOKKOS_INLINE_FUNCTION
int particles_instance::compute_neighbor_cell_index(int cell_index, int dx, int dy, int dz) const {
    int ix = (cell_index % cells_per_dim(0)) + dx;
    int iy = ((cell_index / cells_per_dim(0)) % cells_per_dim(1)) + dy;
    int iz = (cell_index / (cells_per_dim(0) * cells_per_dim(1))) + dz;

    if (ix < 0) ix += cells_per_dim(0);
    if (iy < 0) iy += cells_per_dim(1);
    if (iz < 0) iz += cells_per_dim(2);

    ix %= cells_per_dim(0);
    iy %= cells_per_dim(1);
    iz %= cells_per_dim(2);

    return ix + iy * cells_per_dim(0) + iz * cells_per_dim(0) * cells_per_dim(1);
}

KOKKOS_FUNCTION
void particles_instance::operator()(Tag_populate_cell_list, const int i) const {
    // Calculate the cell index for each particle
    int ix = int(x(i, 0) / cell_size(0));
    int iy = int(x(i, 1) / cell_size(1));
    int iz = int(x(i, 2) / cell_size(2));
    
    // get cell index
    int cell_index = ix + iy * cells_per_dim(0) + iz * cells_per_dim(0) * cells_per_dim(1);
    // increment the cell count for each cell
    // atomic fetch is needed to avoid race conditions
    int pos = Kokkos::atomic_fetch_add(&cell_count(cell_index), 1);

    // finally add the parrrticle index to the cell
    if (pos < cell_list.extent(1)) {
        cell_list(cell_index, pos) = i;
    } else {
        Kokkos::abort("ERROR: cell list dimension not large enouogh for number of particles!");
    }
}

KOKKOS_INLINE_FUNCTION
int particles_instance::compute_cell_index(double x, double y, double z) const {
    int ix = static_cast<int>(x / h_cell_size(0));
    int iy = static_cast<int>(y / h_cell_size(1));
    int iz = static_cast<int>(z / h_cell_size(2));

    return ix + iy * h_cells_per_dim(0) + iz * h_cells_per_dim(0) * h_cells_per_dim(1);
}

double particles_instance::potential_verlet_list() {
    double result = 0.0;
    Kokkos::parallel_reduce("particles-LJ-potential-verlet-list",
        Kokkos::TeamPolicy<Tag_potential_verlet>(N, Kokkos::AUTO), *this, result);
    return 4 * result;
}

KOKKOS_FUNCTION
void particles_instance::operator()(Tag_potential_verlet, const member_type& teamMember, double& V) const {
    const int i = teamMember.league_rank();  // Index of the current cell
    const int type_i = id[i]-1;
    double tmpV = 0.0;
    if (neighbour_count(i) == 0) return;
    // Loop over all neighbours
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, neighbour_count(i)), 
    [=](int j, double& innerV) {
        int particle_j = verlet_list(i,j);
        int type_j = id[particle_j]-1;
        double rx = x(i, 0) - x(particle_j, 0);
        rx -= int(rx * inverse_halved_L[0]) * L[0];
        double r2 = rx * rx;
        if (r2 > cutoff_squared) return;
        double ry = x(i, 1) - x(particle_j, 1);
        ry -= int(ry * inverse_halved_L[1]) * L[1];
        r2 += ry * ry;
        if (r2 > cutoff_squared) return;
        double rz = x(i, 2) - x(particle_j, 2);
        rz -= int(rz * inverse_halved_L[2]) * L[2];
        r2 += rz * rz;
        
        if (r2 < cutoff_squared) {
            //Kokkos::printf("i: %d j: %d r2: %f\n",i,particle_j, sqrt(r2));
            double sr2 = sigma_mat(type_i, type_j) * sigma_mat(type_i, type_j) / r2;
            double sr6 = sr2 * sr2 * sr2;
            innerV += epsilon_mat(type_i, type_j) * sr6 * (sr6 - 1.0);
        }
    }, tmpV);

    Kokkos::single(Kokkos::PerTeam(teamMember), [&]() {
        V += tmpV;
    });
}

void particles_instance::compute_force_verlet_list() {
    Kokkos::deep_copy(f,0);
    typedef Kokkos::TeamPolicy<Tag_force_verlet> team_policy;
    Kokkos::parallel_for("compute_force_verlet_list", team_policy(N, Kokkos::AUTO), *this);
    Kokkos::fence();
}

KOKKOS_FUNCTION
void particles_instance::operator() (Tag_force_verlet, const member_type& teamMember) const {
    const int i = teamMember.league_rank();  // Index of the current cell
    const int type_i = id[i]-1;

    if (neighbour_count(i) == 0) return;
    // Loop over all neighbours
    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, neighbour_count(i)), 
    [=](int j) {
        int particle_j = verlet_list(i,j);
        int type_j = id[particle_j]-1;
        double rx = x(i, 0) - x(particle_j, 0);
        rx -= int(rx * inverse_halved_L[0]) * L[0];
        double r2 = rx * rx;
        if (r2 > cutoff_squared) return;
        double ry = x(i, 1) - x(particle_j, 1);
        ry -= int(ry * inverse_halved_L[1]) * L[1];
        r2 += ry * ry;
        if (r2 > cutoff_squared) return;
        double rz = x(i, 2) - x(particle_j, 2);
        rz -= int(rz * inverse_halved_L[2]) * L[2];
        r2 += rz * rz;
        
        if (r2 < cutoff_squared) {
            double sr2 = sigma_mat(type_i, type_j) * sigma_mat(type_i, type_j) / r2;
            double sr6 = sr2 * sr2 * sr2;
            sr2 = sr6 * (-sr6 + 0.5) / r2;
            double force = 48 * epsilon_mat(type_i, type_j) * sr2;
            //Kokkos::printf("force: %f \n", force);
            // Use atomic add to avoid race conditions
            Kokkos::atomic_add(&f(i, 0), force * rx);
            Kokkos::atomic_add(&f(i, 1), force * ry);
            Kokkos::atomic_add(&f(i, 2), force * rz);

            Kokkos::atomic_add(&f(particle_j, 0), -force * rx);
            Kokkos::atomic_add(&f(particle_j, 1), -force * ry);
            Kokkos::atomic_add(&f(particle_j, 2), -force * rz);
        }
    });
}

void particles_instance::build_verlet_list() {
    // Reset neighbour counts to zero
    Kokkos::deep_copy(neighbour_count, 0);
    Kokkos::parallel_for("populate_verlet_list",
        Kokkos::TeamPolicy<Tag_build_verlet_list>(N, Kokkos::AUTO), *this);
    /* Kokkos::deep_copy(h_neighbour_count, neighbour_count);
    for (int i = 0; i < N; i++) {
        printf("COUNT: %d \n", h_neighbour_count(i));
    }*/
}

KOKKOS_FUNCTION
void particles_instance::operator()(Tag_build_verlet_list, const member_type& teamMember) const {
    const int i = teamMember.league_rank();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, neighbour_count.extent(0)), [=](const int j) {
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
                if (neighbour_count(i) < verlet_list.extent(1)) {
                    verlet_list(i,Kokkos::atomic_fetch_add(&neighbour_count(i),1)) = j;   
                } else {
                    Kokkos::abort("ERROR: Verlet list dimension not large enough for number of neighbours! Increase it manually in the config file with MaxParticles");
                }
            }
        }
    });
    
}