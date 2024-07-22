#ifndef NON_IDENTICAL_PARTICLES_H
#define NON_IDENTICAL_PARTICLES_H

#include <functional>
#include <vector>
#include <atom.hpp>
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>
#include "particles.hpp"
#include "identical_particles.hpp"



class non_identical_particles : public identical_particles {

public:
    const std::string name = "non_identical_particles";
    std::vector<atom_type> atom_type_list;
    std::string parameter_file; ///< name of file containing force field parameters
    std::string start_configuration_file; ///< name of file containing initial configuration
    std::vector<std::vector<double>> epsilon_mat; ///< matrix containing the LJ_epsilon parameters
    std::vector<std::vector<double>> sigma_mat; ///< matrix containing the LJ_sigma parameters
    std::vector<double> coeff_x;    ///< list of coefficients for the position calculation
    double T;   ///< temperature

    // constructor
    non_identical_particles(YAML::Node doc, params_class params);

    void InitX(params_class params);
    void assign_ids(type_id& id);
    void assign_algorithm(YAML::Node& doc);
    void update_positions(const double dt_);
    void mix_parameters(YAML::Node& doc);
    void get_parameters(YAML::Node& parameter_file, std::vector<atom_type>& atom_type_list);
    void compute_coeff_position();

    double potential_all_neighbour_inner_parallel();

    KOKKOS_FUNCTION void operator() (check_in_volume, const int i) const;

    KOKKOS_FUNCTION void operator() (Tag_potential_all_inner_parallel, const member_type& teamMember, double& V) const;
    

    // Destructor
    ~non_identical_particles() {};
};

#endif