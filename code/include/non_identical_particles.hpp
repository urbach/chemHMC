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
    struct Tag_potential_MIC_inner_parallel {};
    struct Tag_force_MIC_inner_parallel {};

    struct Tag_force_AMIC_inner_parallel {};
    struct Tag_potential_AMIC_inner_parallel {};

    const std::string name = "non_identical_particles";
    Kokkos::View<atom_type*> atom_type_list; ///< view containing mass/charge/index of atom types
    Kokkos::View<atom_type*>::HostMirror h_atom_type_list; ///< host mirror of atom_type_list
    std::string parameter_file; ///< name of file containing force field parameters
    std::string start_configuration_file; ///< name of file containing initial configuration
    Kokkos::View<double**> epsilon_mat; ///< matrix containing the LJ_epsilon parameters
    Kokkos::View<double**> sigma_mat; ///< matrix containing the LJ_sigma parameters
    Kokkos::View<double**>::HostMirror h_epsilon_mat; ///< host mirror of epsilon_mat
    Kokkos::View<double**>::HostMirror h_sigma_mat; ///< host mirror of sigma_mat
    Kokkos::View<double*> coeff_x;    ///< list of coefficients for the position calculation
    Kokkos::View<double*>::HostMirror h_coeff_x;    ///< host mirror of coeff_x
    type_id::HostMirror h_id; ///< host mirror of id
    double T;   ///< temperature

    // constructor
    non_identical_particles(YAML::Node doc, params_class params);

    void InitX(params_class params);
    void assign_ids();
    void assign_algorithm(YAML::Node& doc);
    void update_positions(const double dt_);
    void mix_parameters(YAML::Node& doc);
    void get_parameters(YAML::Node& parameter_file, Kokkos::View<atom_type*>& atom_type_list);
    void compute_coeff_position();

    double compute_kinetic_E();

    double potential_all_neighbour_inner_parallel();
    double potential_MICAIP(); ///< all_neighbour_inner_parallel with minimum image convention
    double potential_AMICAIP();

    void compute_force_all_inner_parallel();
    void compute_force_MICAIP();
    void compute_force_AMICAIP();

    KOKKOS_FUNCTION void operator() (kinetic, const int& i, double& sum) const;

    // Misc.
    KOKKOS_FUNCTION void operator() (check_in_volume, const int i) const;

    // Potential calculation
    KOKKOS_FUNCTION void operator() (Tag_potential_all_inner_parallel, const member_type& teamMember, double& V) const;
    KOKKOS_FUNCTION void operator() (Tag_potential_MIC_inner_parallel, const member_type& teamMember, double& V) const;
    KOKKOS_FUNCTION void operator() (Tag_potential_AMIC_inner_parallel, const member_type& teamMember, double& V) const;

    // Force calculation
    KOKKOS_FUNCTION void operator() (Tag_force_inner_parallel, const member_type& teamMember) const;
    KOKKOS_FUNCTION void operator() (Tag_force_MIC_inner_parallel, const member_type& teamMember) const;
    KOKKOS_FUNCTION void operator() (Tag_force_AMIC_inner_parallel, const member_type& teamMember) const;

    // Destructor
    ~non_identical_particles() {};
};

#endif