#include "global.hpp"
#include "particles.hpp"

void particles_instance::init_cell_list(YAML::Node& doc) {
    // calculate size of the cells and allocate the needed views
    cell_size = Kokkos::View<double*>("cell_size", dim_space);
    cells_per_dim = Kokkos::View<int*>("cells_per_dim", dim_space);

    h_cell_size = Kokkos::create_mirror_view(cell_size);
    h_cells_per_dim = Kokkos::create_mirror_view(cells_per_dim);

    for (int dim = 0; dim < 3; ++dim) {
        h_cells_per_dim(dim) = static_cast<int>(L[dim] / cutoff);
    }
    for (int dim = 0; dim < 3; ++dim) {
        h_cell_size(dim) = L[dim] / h_cells_per_dim(dim);
    }
    int total_cells = h_cells_per_dim(0) * h_cells_per_dim(1) * h_cells_per_dim(2);
    if (total_cells < 27) {
        Kokkos::abort("ERROR: Lennard-Jones cutoff distance must be smaller than 1/3 of the smallest cell dimension! aborting...");
    }
    Kokkos::deep_copy(cell_size, h_cell_size);
    Kokkos::deep_copy(cells_per_dim, h_cells_per_dim);
    // Get max particles per cell
    int max_particles_per_cell;
    if (doc["particles"]["MaxParticlesPerCell"]) {
        max_particles_per_cell = check_and_assign_value<int>(doc["particles"], "MaxParticlesPerCell");
    } else {
        // If no user value is supplied, we make a generous estimate
        double cell_volume = h_cell_size(0) * h_cell_size(1) * h_cell_size(2);
        double min_sigma = 10.0;
        for(int i = 0; i < h_atom_type_list.extent(0); ++i) {
            if (h_atom_type_list[i].LJ_sigma < min_sigma) {
                min_sigma = h_atom_type_list[i].LJ_sigma;
            }
        }
        double estimated_atomic_volume = min_sigma * min_sigma * min_sigma;
        max_particles_per_cell = cell_volume / estimated_atomic_volume;
    }
    // Allocate the cell list and cell count views
    cell_list = Kokkos::View<int**>("cell_list",h_cells_per_dim(0) * 
                                    h_cells_per_dim(1) * h_cells_per_dim(2),max_particles_per_cell);
    cell_count = Kokkos::View<int*>("cell_count", h_cells_per_dim(0) * 
                                    h_cells_per_dim(1) * h_cells_per_dim(2));

    h_cell_list = Kokkos::create_mirror_view(cell_list);
    h_cell_count = Kokkos::create_mirror_view(cell_count);

    Kokkos::deep_copy(h_cell_count, 0);
    Kokkos::deep_copy(cell_count, h_cell_count);
    Kokkos::deep_copy(cell_list, 0);
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