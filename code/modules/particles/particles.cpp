#include "particles_type.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "global.hpp"
#include "read_infile.hpp"
#include "particles.hpp"

// constructor
particles_instance::particles_instance(YAML::Node doc, params_class params) :
    particles_type(doc, params) {

    // get interaction parameters from parameter file
    get_parameters(doc, atom_type_list);

    // generate mixed pair parameters
    mix_parameters(doc);

    mass = check_and_assign_value<double>(doc["particles"], "mass");
    beta = check_and_assign_value<double>(doc["particles"], "beta");
    sbeta = sqrt(beta);
    cutoff = check_and_assign_value<double>(doc["particles"], "cutoff");
    cutoff_squared = cutoff * cutoff;
    eps = check_and_assign_value<double>(doc["particles"], "eps");
    sigma = check_and_assign_value<double>(doc["particles"], "sigma");
    name_xyz = check_and_assign_value<std::string>(doc["particles"], "name_xyz");
    start_configuration_file = check_and_assign_value<std::string>(doc, "start_configuration_file");

    // get inverse halved box size (needed for MIC algorithm)
    inverse_halved_L[0] = 2.0/L[0];
    inverse_halved_L[1] = 2.0/L[1];
    inverse_halved_L[2] = 2.0/L[2];

    assign_algorithm(doc);

    std::cout << "particles_type:" << std::endl;
    std::cout << "name:" << name << std::endl;
    std::cout << "mass:" << mass << std::endl;
    std::cout << "beta:" << beta << std::endl;

    compute_coeff_momenta();
    compute_coeff_position();
}

void particles_instance::read_xyz(params_class params) {
    FILE* file = NULL;
    file = fopen(params.start_configuration_file.c_str(), "r");
    if (file == NULL) {
        printf("error in opening file %s\n", params.start_configuration_file.c_str());
        Kokkos::abort("abort");
    }
    int lines = 0;
    char c;

    /* count the newline characters */
    while ((c = fgetc(file)) != EOF) {
        if (c == '\n')
            lines++;
    }
    if (lines % (N + 2) != 0) {
        printf("error: input file %s contains %d lines\n", params.start_configuration_file.c_str(), lines);
        printf("       the number of lines mus be a multiple of N+2=%d\n", N + 2);
        Kokkos::abort("abort");
    }
    int confs = lines / (N + 2);
    printf("confs in input configuration file %d\n", confs);
    // go to last configuration and read it 
    rewind(file);
    int count = 0;
    char id[1000];

    while ((c = fgetc(file)) != EOF) {

        if (c == '\n') {
            count++;
            if (count == (confs - 1) * (N + 2) + 1) {
                for (int i = 0;i < 11;i++) c = fgetc(file);
                fscanf(file, " %d", &params.istart);
                // printf("%d %d\n", params.istart, count);
                // count++;
            }
            if (count == (confs - 1) * (N + 2) + 2) {// if starting of the last conf, count missmatched by fscanf
                break;
            }
        }
    }
    printf("reading last configuration from input file %s\n", params.start_configuration_file.c_str());
    count = 0;
    for (int i = 0; i < N;i++) {
        count += fscanf(file, "%s   %lf   %lf  %lf\n", id, &h_x(i, 0), &h_x(i, 1), &h_x(i, 2));
        // printf("%s   %lf   %lf  %lf\n", id, h_x(i, 0), h_x(i, 1), h_x(i, 2));
    }
    if (name_xyz.compare(id) != 0) {
        printf("name in the xyz file: %s  do not mach the name in the input file: %s\n", id, name_xyz.c_str());
        Kokkos::abort("abort");
    } 
    // printf("%d  %d\n", count, N);
    if (count != N * 4) { Kokkos::abort("error in reading the file"); }
    fclose(file);
    Kokkos::deep_copy(x, h_x);
    // printx();
}

int particles_instance::how_many_confs_xyz(FILE* file) {

    int lines = 0;
    char c;

    /* count the newline characters */
    while ((c = fgetc(file)) != EOF) {
        if (c == '\n')
            lines++;
    }
    if (lines % (N + 2) != 0) {
        printf("error: xyz file contains %d lines\n", lines);
        printf("       the number of lines mus be a multiple of N+2=%d\n", N + 2);
        Kokkos::abort("abort");
    }
    int confs = lines / (N + 2);
    printf("confs in input configuration file %d\n", confs);
    rewind(file);
    return confs;
}

void particles_instance::read_next_confs_xyz(FILE* file) {
    int count = 0;
    char id[1000];
    char c;
    while ((c = fgetc(file)) != EOF) {
        if (c == '\n') {
            for (int i = 0;i < 11;i++) c = fgetc(file);
            int tmp;
            count += fscanf(file, " %d", &tmp);
            break;
        }
    }
    while ((c = fgetc(file)) != EOF) { if (c == '\n') break; }
    for (int i = 0; i < N;i++) {
        count += fscanf(file, "%s   %lf   %lf  %lf\n", id, &h_x(i, 0), &h_x(i, 1), &h_x(i, 2));
        // printf("%s   %lf   %lf  %lf\n", id, h_x(i, 0), h_x(i, 1), h_x(i, 2));
    }
    if (name_xyz.compare(id) != 0) {
        printf("name in the xyz file: %s  do not mach the name in the input file: %s\n", id, name_xyz.c_str());
        Kokkos::abort("abort");
    }
    // printf("%d  %d\n", count, N);
    if (count != N * 4 + 1) { Kokkos::abort("error in reading the file"); }
    Kokkos::deep_copy(x, h_x);
    // printx();
}

void particles_instance::get_parameters(YAML::Node& doc, Kokkos::View<atom_type*>& atom_type_list) {
    parameter_file = check_and_assign_value<std::string>(doc, "parameter_file");

    std::ifstream infile(parameter_file);
    std::string line;
    std::getline(infile, line); // Skip the first line
    std::vector<atom_type> temp_atom_type_list;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string label;
        double mass, charge, epsilon, sigma;
        int index;

        iss >> index >> label >> mass >> charge >> epsilon >> sigma;

        // Create an atom_type instance and add it to the vector
        atom_type atom(label.c_str(), mass, charge, index, epsilon, sigma);
        temp_atom_type_list.push_back(atom);
    }
    h_atom_type_list = Kokkos::create_mirror_view(Kokkos::View<atom_type*>("atom_type_list", temp_atom_type_list.size()));
    for (size_t i = 0; i < temp_atom_type_list.size(); ++i) {
        h_atom_type_list(i) = temp_atom_type_list[i];
    }
    atom_type_list = Kokkos::View<atom_type*>("atom_type_list", temp_atom_type_list.size());

    Kokkos::deep_copy(atom_type_list, h_atom_type_list);
}

void particles_instance::mix_parameters(YAML::Node& doc) {

    // assign correct size to parameter matrices
    int num_atom_types = atom_type_list.extent(0);
    sigma_mat = Kokkos::View<double**>("sigma_mat", num_atom_types, num_atom_types);
    epsilon_mat = Kokkos::View<double**>("epsilon_mat", num_atom_types, num_atom_types);

    // Initialize host mirrors
    h_sigma_mat = Kokkos::create_mirror_view(sigma_mat);
    h_epsilon_mat = Kokkos::create_mirror_view(epsilon_mat);


    // Fill known diagonal elements
    for(int i = 0; i < h_atom_type_list.extent(0); ++i) {
        h_epsilon_mat(i, i) = h_atom_type_list(i).LJ_epsilon;
        h_sigma_mat(i, i) = h_atom_type_list(i).LJ_sigma;
    }

    // Apply Lorentz-Berthelot mixing rules
    for(int i = 0; i < h_atom_type_list.extent(0); ++i) {
        for(int j = i + 1; j < h_atom_type_list.extent(0); ++j) {
            double epsilon = sqrt(h_epsilon_mat(i, i) * h_epsilon_mat(j, j)); // Berthelots rule
            double sigma = 0.5 * (h_sigma_mat(i, i) + h_sigma_mat(j, j)); // Lorentz rule
            h_epsilon_mat(i, j) = epsilon;
            h_epsilon_mat(j, i) = epsilon;
            h_sigma_mat(i, j) = sigma;
            h_sigma_mat(j, i) = sigma;
        }

    }

    // Copy the updated data to device memory
    Kokkos::deep_copy(sigma_mat, h_sigma_mat);
    Kokkos::deep_copy(epsilon_mat, h_epsilon_mat);
}

void particles_instance::assign_algorithm(YAML::Node& doc) {
    algorithm = check_and_assign_value<std::string>(doc["particles"], "algorithm");
    printf("ALGORITHM: %s \n", algorithm.c_str());
    if (algorithm.compare("all_neighbour") == 0) {
        printf("selected algorithm: %s is not implemented for non identical particles\n", algorithm.c_str());
        Kokkos::abort("aborting");
    }
    else if (algorithm.compare("all_neighbour_inner_parallel") == 0) {
        potential_strategy = std::bind(&particles_instance::potential_all_neighbour_inner_parallel, this);
        potential_without_binning_strategy = std::bind(&particles_instance::potential_all_neighbour_inner_parallel, this);
        force_strategy = std::bind(&particles_instance::compute_force_all_inner_parallel, this);
    }
    else if (algorithm.compare("MICAIP") == 0) {
        potential_strategy = std::bind(&particles_instance::potential_MICAIP, this);
        potential_without_binning_strategy = std::bind(&particles_instance::potential_MICAIP, this);
        force_strategy = std::bind(&particles_instance::compute_force_MICAIP, this);
    }
    else if (algorithm.compare("AMIC") == 0) {
        potential_strategy = std::bind(&particles_instance::potential_AMICAIP, this);
        potential_without_binning_strategy = std::bind(&particles_instance::potential_AMICAIP, this);
        force_strategy = std::bind(&particles_instance::compute_force_AMICAIP, this);
    }
    else if (algorithm.compare("cell_list") == 0) {
        particles_instance::init_cell_list(doc);
        potential_strategy = std::bind(&particles_instance::potential_cell_list, this);
        potential_without_binning_strategy = std::bind(&particles_instance::potential_cell_list, this);
        force_strategy = std::bind(&particles_instance::compute_force_cell_list, this);
    }
    else if (algorithm.compare("verlet_list") == 0) {
        particles_instance::init_verlet_list(doc);
        potential_strategy = std::bind(&particles_instance::potential_verlet_list, this);
        potential_without_binning_strategy = std::bind(&particles_instance::potential_verlet_list, this);
        force_strategy = std::bind(&particles_instance::compute_force_verlet_list, this);
    }
    else if (algorithm.compare("parallel_binning") == 0) {
        printf("selected algorithm: %s is not implemented for non identical particles\n", algorithm.c_str());
        Kokkos::abort("aborting");
    }
    else if (algorithm.compare("quick_sort") == 0) {
        printf("selected algorithm: %s is not implemented for non identical particles\n", algorithm.c_str());
        Kokkos::abort("aborting");
    }
    else {
        printf("selected algorithm: %s is not a valid algorithm\n", algorithm.c_str());
        Kokkos::abort("aborting");
    }
}

void particles_instance::init_verlet_list(YAML::Node& doc) {
    int max_neighbors;
    if (doc["particles"]["MaxNeighbors"]) {
        max_neighbors = check_and_assign_value<int>(doc["particles"], "MaxNeighbors");
    } else {
        max_neighbors = 50; // A typical number; adjust as needed.
    }
    
    verlet_list = Kokkos::View<int**>("verlet_list", N, max_neighbors);
    h_verlet_list = Kokkos::create_mirror_view(verlet_list);

    Kokkos::deep_copy(h_verlet_list, 0);
    Kokkos::deep_copy(verlet_list, h_verlet_list);

    neighbour_count = Kokkos::View<int*>("neighbour_count", N);
    h_neighbour_count = Kokkos::create_mirror_view(neighbour_count);
}

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

void particles_instance::assign_ids() {
    // this function reads in the atom types from the start_configuration_file
    // and assigns each an atom_type_id defined in the atom_type_list.

    std::ifstream infile(start_configuration_file);
    if (!infile) {
        throw std::runtime_error("Unable to open parameter file: " + start_configuration_file);
    }

    std::string line;
    // get number of particles from first line
    std::getline(infile, line);
    std::istringstream iss(line);
    int N_particles;
    iss >> N_particles;

    // skip comment line 
    std::getline(infile, line);
    // assign all ids
    for (int i = 0; i < N_particles; i++) {
        // get type from current line
        std::getline(infile, line);
        std::istringstream iss(line);
        std::string type;
        iss >> type;
        
        // find matching id and assign it
        for (int j = 0; j < h_atom_type_list.extent(0); j++) {
            if(type == h_atom_type_list[j].label) {
                h_id[i] = h_atom_type_list[j].type_index;
                break;
            }
        }
    }
}

void particles_instance::InitX(params_class params) {
    x = type_x("x", N);
    // create_mirror() will always allocate a new view,
    // create_mirror_view() will only create a new view if the original one is not in HostSpace
    h_x = Kokkos::create_mirror(x);
    p = type_p("p", N);
    f = type_f("f", N);

    // save atom_type id for each particle
    id = type_id("id", N);
    h_id = Kokkos::create_mirror(id);
    assign_ids();
    Kokkos::deep_copy(id, h_id);

    if (params.StartCondition == "read") {
        read_xyz(params);
    }
    else {
        Kokkos::abort("StartCondition for particles must be 'read'");
    }
    
    Kokkos::deep_copy(h_x, x);
    Kokkos::parallel_for("volume_check", Kokkos::RangePolicy<check_in_volume>(0, N), *this);
    Kokkos::fence();
    printf("particle initialized\n");
}

void particles_instance::compute_coeff_momenta() {
    coeff_p = beta;
}

void particles_instance::compute_coeff_position() {

    // inititalize device and host views
    coeff_x = Kokkos::View<double*>("coeff_x",h_atom_type_list.extent(0));
    h_coeff_x = Kokkos::create_mirror_view(coeff_x);
    
    //Since we have different particles we need to compute one coefficient for each type
    for(int i = 0;i < h_atom_type_list.extent(0);i++) {
        h_coeff_x[i] = beta / (h_atom_type_list[i].mass);
    }

    //copy to device
    Kokkos::deep_copy(coeff_x, h_coeff_x);
}

double particles_instance::compute_kinetic_E() {
    double K = 0;
    Kokkos::parallel_reduce("identical-particles-LJ-kinetic-E", Kokkos::RangePolicy<kinetic>(0, N), *this, K);
    return K;
}

KOKKOS_FUNCTION
void particles_instance::operator() (kinetic, const int& i, double& sum) const {
    sum += (p(i, 0) * p(i, 0) + p(i, 1) * p(i, 1) + p(i, 2) * p(i, 2)) / (2 * atom_type_list[id[i]-1].mass);
};

class functor_update_pos {
public:
    const double dt;
    Kokkos::View<double*> c;
    type_x x;
    type_id id;
    type_const_p p;
    const double L[dim_space];
    functor_update_pos(double dt_, Kokkos::View<double*> c_, type_x& x_, type_p& p_, type_id& id_,const double L_[]) : dt(dt_), c(c_), x(x_), p(p_), id(id_),
        L{ L_[0], L_[1], L_[2] } {
    };

    KOKKOS_FUNCTION
        void operator() (const int i) const {
        for (int dir = 0; dir < 3; dir++) {
            x(i, dir) += dt * c[id[i]-1] * p(i, dir);
            // apply  periodic boundary condition
            x(i, dir) -= L[dir] * floor(x(i, dir) / L[dir]);
        }
    };
};
void particles_instance::update_positions(const double dt_) {
    Kokkos::parallel_for("update_position", Kokkos::RangePolicy(0, N), functor_update_pos(dt_, coeff_x, x, p, id, L));
}

class functor_update_momenta {
public:
    const double dt;
    const double c;
    type_p p;
    type_const_f f;
    functor_update_momenta(double dt_, double c_, type_p& p_, type_f& f_) : dt(dt_), c(c_), p(p_), f(f_) {};

    KOKKOS_FUNCTION
        void operator() (const int i) const {
        p(i, 0) -= dt * c * f(i, 0);
        p(i, 1) -= dt * c * f(i, 1);
        p(i, 2) -= dt * c * f(i, 2);
    };
};
void particles_instance::update_momenta(const double dt_) {
    compute_force();
    Kokkos::parallel_for("update_momenta", Kokkos::RangePolicy(0, N), functor_update_momenta(dt_, coeff_p, p, f));
}

KOKKOS_FUNCTION
void particles_instance::operator() (check_in_volume, const int i) const {
    for (int dir = 0; dir < dim_space;dir++) {
        if (x(i, dir) < 0 || x(i, dir) >= L[dir]) {
            printf("error: particle position x(%d, %d)= %g  outside the box of length %g\n", i, dir, x(i, dir), L[dir]);
            Kokkos::abort("aborting");
        }
    }
};

KOKKOS_FUNCTION
void particles_instance::operator() (cold, const int i) const {
    double N3 = pow(N, 1. / 3.);
    int iz = (int)i / (N3 * N3);
    int iy = (int)(i - iz * N3 * N3) / (N3);
    int ix = (int)(i - iz * N3 * N3 - iy * N3);

    x(i, 0) = L[0] * (ix - N3 * floor(ix / N3)) / (N3 + 1);
    x(i, 1) = L[1] * (iy - N3 * floor(iy / N3)) / (N3 + 1);
    x(i, 2) = L[2] * (iz - N3 * floor(iz / N3)) / (N3 + 1);
};

KOKKOS_FUNCTION
void particles_instance::operator() (hot, const int i) const {
    gen_type rgen = rand_pool.get_state(i);
    x(i, 0) = rgen.drand() * L[0];
    x(i, 1) = rgen.drand() * L[1];
    x(i, 2) = rgen.drand() * L[2];
    rand_pool.free_state(rgen);
};

// since we are using the hostMirror to store the starting point we don't whant to 
// deep_copy it here 
void particles_instance::print_xyz(params_class params, int traj, double K, double V) {
    fprintf(params.fileout, "     %d\n", N);
    fprintf(params.fileout, "trajectory= %d  kinetic_energy= %.12g  potential= %.12g\n", traj, K, V);
    for (int i = 0; i < N; i++)
        fprintf(params.fileout, "%s  %-20.12g %-20.12g %-20.12g\n", name_xyz.c_str(), h_x(i, 0), h_x(i, 1), h_x(i, 2));
}

void particles_instance::hb() {
    Kokkos::parallel_for("hb_momenta", Kokkos::RangePolicy<hbTag>(0, N), *this);
}

KOKKOS_FUNCTION
void particles_instance::operator() (hbTag, const int i) const {
    gen_type rgen = rand_pool.get_state(i);
    // we need to divide by sqrt(2) in order to have exp(-p^2)
    // normal() produced distribution exp(-p^2/2)
    p(i, 0) = rgen.normal() * Kokkos::sqrt(mass / beta);
    p(i, 1) = rgen.normal() * Kokkos::sqrt(mass / beta);
    p(i, 2) = rgen.normal() * Kokkos::sqrt(mass / beta);
    rand_pool.free_state(rgen);
}