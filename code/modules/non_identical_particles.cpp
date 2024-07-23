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

void non_identical_particles::get_parameters(YAML::Node& doc, std::vector<atom_type>& atom_type_list) {

    parameter_file = check_and_assign_value<std::string>(doc, "parameter_file");

    std::ifstream infile(parameter_file);
    if (!infile) {
        throw std::runtime_error("Unable to open parameter file: " + parameter_file);
    }

    std::string line;
    // Skip the header line
    std::getline(infile, line);

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string label;
        double mass, charge, epsilon, sigma;
        int index;

        iss >> index >> label >> mass >> charge >> epsilon >> sigma;
        //label = trim(label); // Trim any extraneous whitespace from label

        // Create an atom_type instance and add it to the vector
        atom_type atom(label, mass, charge, index, epsilon, sigma);
        atom_type_list.push_back(atom);
    }
}

void non_identical_particles::mix_parameters(YAML::Node& doc) {

    // resize parameter matrices to correct size
    epsilon_mat.resize(atom_type_list.size(), std::vector<double>(atom_type_list.size()));
    sigma_mat.resize(atom_type_list.size(), std::vector<double>(atom_type_list.size()));

    // fill known diagonal elements
    for(int i = 0; i<atom_type_list.size(); i++) {
        epsilon_mat[i][i]=atom_type_list[i].LJ_epsilon;
        sigma_mat[i][i]=atom_type_list[i].LJ_sigma;
    }

    // apply Lorentz-Berthelot mixing rules
    double sigma;
    double epsilon;
    for(int i = 0; i < atom_type_list.size(); i++)  {
        for(int j = i+1; j < atom_type_list.size(); j++) {
            epsilon = sqrt(epsilon_mat[i][i]*epsilon_mat[j][j]); //Berthelots rule
            sigma = 0.5*(sigma_mat[i][i]+sigma_mat[j][j]); // Lorentz rule
            epsilon_mat[i][j] = epsilon;
            epsilon_mat[j][i] = epsilon;
            sigma_mat[i][j] = sigma;
            sigma_mat[j][i] = sigma;
        }
    }
}

void non_identical_particles::assign_algorithm(YAML::Node& doc) {
    algorithm = check_and_assign_value<std::string>(doc["particles"], "algorithm");
    if (algorithm.compare("all_neighbour") == 0) {
        potential_strategy = std::bind(&identical_particles::potential_all_neighbour, this);
        potential_without_binning_strategy = std::bind(&identical_particles::potential_all_neighbour, this);
        force_strategy = std::bind(&identical_particles::compute_force_all, this);
    }
    else if (algorithm.compare("all_neighbour_inner_parallel") == 0) {
        potential_strategy = std::bind(&non_identical_particles::potential_all_neighbour_inner_parallel, this);
        potential_without_binning_strategy = std::bind(&non_identical_particles::potential_all_neighbour_inner_parallel, this);
        force_strategy = std::bind(&non_identical_particles::compute_force_all_inner_parallel, this);
    }
    else if (algorithm.compare("binning_serial") == 0) {
        binning_geometry_strategy = std::bind(&identical_particles::cutoff_binning, this);
        binning_geometry();
        binning_strategy = std::bind(&identical_particles::serial_binning, this);
        serial_binning_init();
        potential_strategy = std::bind(&identical_particles::potential_binning, this);
        potential_without_binning_strategy = std::bind(&identical_particles::potential_with_binning_set, this);
        force_strategy = std::bind(&identical_particles::compute_force_binning, this);
    }
    else if (algorithm.compare("parallel_binning") == 0) {
        binning_geometry_strategy = std::bind(&identical_particles::cutoff_binning, this);
        binning_geometry();
        binning_strategy = std::bind(&identical_particles::parallel_binning, this);
        parallel_binning_init();
        potential_strategy = std::bind(&identical_particles::potential_binning, this);
        potential_without_binning_strategy = std::bind(&identical_particles::potential_with_binning_set, this);
        force_strategy = std::bind(&identical_particles::compute_force_binning, this);
    }
    else if (algorithm.compare("quick_sort") == 0) {
        binning_geometry_strategy = std::bind(&identical_particles::cutoff_binning, this);
        binning_geometry();
        binning_strategy = std::bind(&identical_particles::create_quick_sort, this);
        quick_sort_init();
        potential_strategy = std::bind(&identical_particles::potential_binning, this);
        potential_without_binning_strategy = std::bind(&identical_particles::potential_with_binning_set, this);
        force_strategy = std::bind(&identical_particles::compute_force_binning, this);
    }
    else {
        printf("selected algorithm: %s is not a valid algorithm\n", algorithm.c_str());
        Kokkos::abort("aborting");
    }
}

void non_identical_particles::assign_ids(type_id& id) {
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
        for (int j = 0; j < atom_type_list.size(); j++) {
            if(type == atom_type_list[j].label) {
                id[i] = atom_type_list[j].type_index;
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
    assign_ids(id);
    
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
    //Since we have different particles we need to compute one coefficient for each type
    for(int i = 0;i < atom_type_list.size();i++) {
        coeff_x.push_back(beta / (atom_type_list[i].mass));
    }
}

class functor_update_pos_non_identical {
public:
    const double dt;
    std::vector<double> c;
    type_x x;
    type_id id;
    type_const_p p;
    const double L[dim_space];
    functor_update_pos_non_identical(double dt_, std::vector<double> c_, type_x& x_, type_p& p_, type_id& id_,const double L_[]) : dt(dt_), c(c_), x(x_), p(p_), id(id_),
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

double non_identical_particles::potential_all_neighbour_inner_parallel() {
    double result;
    Kokkos::parallel_reduce("identical_particles-LJ-potential-all-inner-parallel",
        Kokkos::TeamPolicy<Tag_potential_all_inner_parallel>(N, Kokkos::AUTO), *this, result);
    // 2 *eps instead of 4 *eps because we count the couples i,j twice
    return 2 * result;
}


KOKKOS_FUNCTION
void non_identical_particles::operator() (Tag_potential_all_inner_parallel, const member_type& teamMember, double& V) const {
    const int i = teamMember.league_rank();
    double tmpV;
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
                            double sr2 = sigma_mat[type_i][type_j] * sigma_mat[type_i][type_j] / r2;
                            double sr6 = sr2 * sr2 * sr2;
                            innerV += epsilon_mat[type_i][type_j] * sr6 * (sr6 - 1.0);
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
    Kokkos::parallel_for("identical_particles-LJ-force-all-inner-parall", team_policy(N, Kokkos::AUTO), *this);
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
                            double sr2 = sigma_mat[type_i][type_j] * sigma_mat[type_i][type_j] / r2;
                            double sr6 = sr2 * sr2 * sr2;
                            sr2 = sr6 * (-sr6 + 0.5) / r2;
                            innerfv.the_array[0] += epsilon_mat[type_i][type_j] * sr2 * (x(i, 0) - (x(j, 0) + bx * L[0]));
                            innerfv.the_array[1] += epsilon_mat[type_i][type_j] * sr2 * (x(i, 1) - (x(j, 1) + by * L[1]));
                            innerfv.the_array[2] += epsilon_mat[type_i][type_j] * sr2 * (x(i, 2) - (x(j, 2) + bz * L[2]));
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