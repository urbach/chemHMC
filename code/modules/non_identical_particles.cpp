#include "particles.hpp"

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
#include "identical_particles.hpp"
#include "non_identical_particles.hpp"
#include "binning.hpp"

// constructor
non_identical_particles::non_identical_particles(YAML::Node doc, params_class params) :
    identical_particles(doc, params) {

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

    if (doc["particles"]["RDF"]) {
        size_bRDF = check_and_assign_value<double>(doc["particles"]["RDF"], "size_bin");
        LmaxRDF = check_and_assign_value<double>(doc["particles"]["RDF"], "Lmax");
        filename_RDF = check_and_assign_value<std::string>(doc["particles"]["RDF"], "output_file");
        NbRDF = (int)(LmaxRDF / size_bRDF);
        printf("RDF: Lmax=%g  size_bin=%g  N=%d  \n", LmaxRDF, size_bRDF, NbRDF);
        RDF = t_RDF("RDF", NbRDF);
        h_RDF = Kokkos::create_mirror(RDF);
    }
}

void non_identical_particles::get_parameters(YAML::Node& doc, Kokkos::View<atom_type*>& atom_type_list) {
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

void non_identical_particles::mix_parameters(YAML::Node& doc) {

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

void non_identical_particles::assign_algorithm(YAML::Node& doc) {
    algorithm = check_and_assign_value<std::string>(doc["particles"], "algorithm");
    printf("ALGORITHM: %s", algorithm.c_str());
    if (algorithm.compare("all_neighbour") == 0) {
        printf("selected algorithm: %s is not implemented for non identical particles\n", algorithm.c_str());
        Kokkos::abort("aborting");
    }
    else if (algorithm.compare("all_neighbour_inner_parallel") == 0) {
        potential_strategy = std::bind(&non_identical_particles::potential_all_neighbour_inner_parallel, this);
        potential_without_binning_strategy = std::bind(&non_identical_particles::potential_all_neighbour_inner_parallel, this);
        force_strategy = std::bind(&non_identical_particles::compute_force_all_inner_parallel, this);
    }
    else if (algorithm.compare("MICAIP") == 0) {
        potential_strategy = std::bind(&non_identical_particles::potential_MICAIP, this);
        potential_without_binning_strategy = std::bind(&non_identical_particles::potential_MICAIP, this);
        force_strategy = std::bind(&non_identical_particles::compute_force_MICAIP, this);
    }
    else if (algorithm.compare("AMIC") == 0) {
        potential_strategy = std::bind(&non_identical_particles::potential_AMICAIP, this);
        potential_without_binning_strategy = std::bind(&non_identical_particles::potential_AMICAIP, this);
        force_strategy = std::bind(&non_identical_particles::compute_force_AMICAIP, this);
    }
    else if (algorithm.compare("binning_serial") == 0) {
        printf("selected algorithm: %s is not implemented for non identical particles\n", algorithm.c_str());
        Kokkos::abort("aborting");
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

void non_identical_particles::assign_ids() {
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

void non_identical_particles::InitX(params_class params) {
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
        Kokkos::abort("StartCondition for non_identical_particles must be 'read'");
    }
    
    Kokkos::deep_copy(h_x, x);
    Kokkos::parallel_for("volume_check", Kokkos::RangePolicy<check_in_volume>(0, N), *this);
    Kokkos::fence();
    printf("particle initialized\n");
}

void non_identical_particles::compute_coeff_position() {

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

double non_identical_particles::compute_kinetic_E() {
    double K = 0;
    Kokkos::parallel_reduce("identical-particles-LJ-kinetic-E", Kokkos::RangePolicy<kinetic>(0, N), *this, K);
    return K;
}

KOKKOS_FUNCTION
void non_identical_particles::operator() (kinetic, const int& i, double& sum) const {
    sum += (p(i, 0) * p(i, 0) + p(i, 1) * p(i, 1) + p(i, 2) * p(i, 2)) / (2 * atom_type_list[id[i]-1].mass);
};

class functor_update_pos_non_identical {
public:
    const double dt;
    Kokkos::View<double*> c;
    type_x x;
    type_id id;
    type_const_p p;
    const double L[dim_space];
    functor_update_pos_non_identical(double dt_, Kokkos::View<double*> c_, type_x& x_, type_p& p_, type_id& id_,const double L_[]) : dt(dt_), c(c_), x(x_), p(p_), id(id_),
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
void non_identical_particles::update_positions(const double dt_) {
    Kokkos::parallel_for("update_position", Kokkos::RangePolicy(0, N), functor_update_pos_non_identical(dt_, coeff_x, x, p, id, L));
}

KOKKOS_FUNCTION
void non_identical_particles::operator() (check_in_volume, const int i) const {
    for (int dir = 0; dir < dim_space;dir++) {
        if (x(i, dir) < 0 || x(i, dir) >= L[dir]) {
            printf("error: particle position x(%d, %d)= %g  outside the box of length %g\n", i, dir, x(i, dir), L[dir]);
            Kokkos::abort("aborting");
        }
    }
};

double non_identical_particles::potential_MICAIP() {
    double result;
    Kokkos::parallel_reduce("non_identical_particles-LJ-potential-MICAIP",
        Kokkos::TeamPolicy<Tag_potential_MIC_inner_parallel>(N, Kokkos::AUTO), *this, result);
    // 2 *eps instead of 4 *eps because we count the couples i,j twice
    return 2 * result;
}

KOKKOS_FUNCTION
void non_identical_particles::operator() (Tag_potential_MIC_inner_parallel, const member_type& teamMember, double& V) const {
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

void non_identical_particles::compute_force_MICAIP() {
    typedef Kokkos::TeamPolicy<Tag_force_MIC_inner_parallel>  team_policy;
    Kokkos::parallel_for("non_identical_particles-LJ-force-MICAIP", team_policy(N, Kokkos::AUTO), *this);
}

double non_identical_particles::potential_AMICAIP() {
    double result;
    Kokkos::parallel_reduce("non_identical_particles-LJ-potential-AMICAIP",
        Kokkos::TeamPolicy<Tag_potential_AMIC_inner_parallel>(N, Kokkos::AUTO), *this, result);
    return 4 * result;
}

KOKKOS_FUNCTION
void non_identical_particles::operator() (Tag_potential_AMIC_inner_parallel, const member_type& teamMember, double& V) const {
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
void non_identical_particles::operator() (Tag_force_MIC_inner_parallel, const member_type& teamMember) const {
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

void non_identical_particles::compute_force_AMICAIP() {
    typedef Kokkos::TeamPolicy<Tag_force_AMIC_inner_parallel>  team_policy;
    Kokkos::parallel_for("non_identical_particles-LJ-force-AMICAIP", team_policy(N, Kokkos::AUTO), *this);
}

KOKKOS_FUNCTION
void non_identical_particles::operator() (Tag_force_AMIC_inner_parallel, const member_type& teamMember) const {
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

double non_identical_particles::potential_all_neighbour_inner_parallel() {
    double result;
    Kokkos::parallel_reduce("non_identical_particles-LJ-potential-all-inner-parallel",
        Kokkos::TeamPolicy<Tag_potential_all_inner_parallel>(N, Kokkos::AUTO), *this, result);
    // 2 *eps instead of 4 *eps because we count the couples i,j twice
    return 2 * result;
}

KOKKOS_FUNCTION
void non_identical_particles::operator() (Tag_potential_all_inner_parallel, const member_type& teamMember, double& V) const {
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

void non_identical_particles::compute_force_all_inner_parallel() {
    typedef Kokkos::TeamPolicy<Tag_force_inner_parallel>  team_policy;
    Kokkos::parallel_for("non_identical_particles-LJ-force-all-inner-parallel", team_policy(N, Kokkos::AUTO), *this);
}

KOKKOS_FUNCTION
void non_identical_particles::operator() (Tag_force_inner_parallel, const member_type& teamMember) const {
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