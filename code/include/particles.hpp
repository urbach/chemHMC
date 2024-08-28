#ifndef particles_H
#define particles_H

#include <functional>
#include <vector>
#include <atom.hpp>
#include <iostream>
#include <iomanip>
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>
#include "particles_type.hpp"

// Structure to hold bond type data
struct BondType {
    int type;
    double k;    // Force constant
    double r0;   // Equilibrium distance
};

// Structure to hold angle type data
struct AngleType {
    int type;
    double k;    // Force constant
    double theta0; // Equilibrium angle (in degrees)
};

// Structure to hold bond data
struct Bond {
    int id;
    int type;
    int atom1;
    int atom2;
};

// Structure to hold angle data
struct Angle {
    int id;
    int type;
    int atom1;
    int atom2;
    int atom3;
};

class particles_instance : public particles_type {

public:
    typedef Kokkos::TeamPolicy<>::member_type  member_type;

    double mass;
    double beta;
    double sbeta;// sqrt(beta)
    double sigma;
    double eps;
    double cutoff;
    double cutoff_squared;
    std::string name_xyz;

    const std::string name = "particles";
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


    // cell_list related stuff
    Kokkos::View<int*> cells_per_dim;
    Kokkos::View<int*>::HostMirror h_cells_per_dim;
    Kokkos::View<double*> cell_size;   // Size of each cell
    Kokkos::View<double*>::HostMirror h_cell_size;
    Kokkos::View<int**> cell_list;
    Kokkos::View<int**>::HostMirror h_cell_list;
    Kokkos::View<int*> cell_count; // Store the number of particles in each cell
    Kokkos::View<int*>::HostMirror h_cell_count;

    // verlet_list related stuff
    Kokkos::View<int*> neighbour_count;
    Kokkos::View<int*>::HostMirror h_neighbour_count;
    Kokkos::View<int**> verlet_list;
    Kokkos::View<int**>::HostMirror h_verlet_list;


    // bond related stuff
    Kokkos::View<int*[3]> bond_list;
    Kokkos::View<int*[3]>::HostMirror h_bond_list;
    Kokkos::View<double*[2]> bond_parameters;
    Kokkos::View<double*[2]>::HostMirror h_bond_parameters;


    // constructor
    particles_instance(YAML::Node doc, params_class params);


    // file interaction
    void print_xyz(params_class params, int traj, double K, double V) override;
    void read_xyz(params_class params) override;
    int how_many_confs_xyz(FILE* file) override;
    void read_next_confs_xyz(FILE* file) override;
    void get_parameters(YAML::Node& parameter_file, Kokkos::View<atom_type*>& atom_type_list);


    // initialization related stuff
    struct cold {};
    struct hot {};
    struct check_in_volume {};

    void InitX(params_class params) override;
    double get_beta() { return beta; };
    void assign_ids();
    void assign_algorithm(YAML::Node& doc);
    void mix_parameters(YAML::Node& doc);
    void compute_coeff_position();
    void compute_coeff_momenta();

    KOKKOS_FUNCTION void operator() (cold, const int i) const;
    KOKKOS_FUNCTION void operator() (hot, const int i) const;
    KOKKOS_FUNCTION void operator() (check_in_volume, const int i) const;
    

    // integrator related stuff
    struct hbTag {};

    void hb() override;
    void update_positions(const double dt_) override;
    void update_momenta(const double dt_) override;

    KOKKOS_FUNCTION void operator() (hbTag, const int i) const;
    
    // kinetic energy calculation
    struct kinetic {};

    double compute_kinetic_E() override;

    KOKKOS_FUNCTION void operator() (kinetic, const int& i, double& sum) const;


    // potential energy calculation
    struct Tag_potential_all_inner_parallel {};
    struct Tag_potential_MIC_inner_parallel {};
    struct Tag_potential_AMIC_inner_parallel {};
    struct Tag_potential_cell {};
    struct Tag_potential_verlet {};

    std::function<double()>  potential_strategy;
    double potential_all_neighbour_inner_parallel();
    double potential_MICAIP(); ///< all_neighbour_inner_parallel with minimum image convention
    double potential_AMICAIP();
    double potential_cell_list();
    double potential_verlet_list();
    double compute_potential() override {
        return potential_strategy();
    };
    std::function<double()>  potential_without_binning_strategy;
    double evaluate_potential() override {
        return potential_without_binning_strategy();
    };

    KOKKOS_FUNCTION void operator() (Tag_potential_all_inner_parallel, const member_type& teamMember, double& V) const;
    KOKKOS_FUNCTION void operator() (Tag_potential_MIC_inner_parallel, const member_type& teamMember, double& V) const;
    KOKKOS_FUNCTION void operator() (Tag_potential_AMIC_inner_parallel, const member_type& teamMember, double& V) const;
    KOKKOS_FUNCTION void operator() (Tag_potential_cell, const member_type& teamMember, double& V) const;
    KOKKOS_FUNCTION void operator() (Tag_potential_verlet, const member_type& teamMember, double& V) const;


    void minimize_energy(YAML::Node& doc) override;


    // force calculation
    struct force {};
    struct Tag_force_inner_parallel {};
    struct Tag_force_MIC_inner_parallel {};
    struct Tag_force_AMIC_inner_parallel {};
    struct Tag_force_cell {};
    struct Tag_force_verlet {};

    std::function<void()>  force_strategy;
    void compute_force_all_inner_parallel();
    void compute_force_MICAIP();
    void compute_force_AMICAIP();
    void compute_force_cell_list();
    void compute_force_verlet_list();
    void compute_force() override {
        force_strategy();
    };

    KOKKOS_FUNCTION void operator() (Tag_force_inner_parallel, const member_type& teamMember) const;
    KOKKOS_FUNCTION void operator() (Tag_force_MIC_inner_parallel, const member_type& teamMember) const;
    KOKKOS_FUNCTION void operator() (Tag_force_AMIC_inner_parallel, const member_type& teamMember) const;
    KOKKOS_FUNCTION void operator() (Tag_force_cell, const member_type& teamMember) const;
    KOKKOS_FUNCTION void operator() (Tag_force_verlet, const member_type& teamMember) const;

    // Cell list
    void init_cell_list(YAML::Node& doc);
    void populate_cell_list();
    int compute_cell_index(double x, double y, double z) const;

    struct Tag_populate_cell_list {};

    KOKKOS_FUNCTION void operator()(Tag_populate_cell_list, const int i) const;
    KOKKOS_INLINE_FUNCTION int compute_neighbor_cell_index(int cell_index, int dx, int dy, int dz) const;


    // Verlet list
    void init_verlet_list(YAML::Node& doc);
    void build_verlet_list() override;

    struct Tag_build_verlet_list {};

    KOKKOS_FUNCTION void operator() (Tag_build_verlet_list, const member_type& teamMember) const;


    // Ewald sum
    double ewald_alpha;
    int k_max;
    int ewald_n_max;
    Kokkos::View<double*> charge;
    Kokkos::View<double*>::HostMirror h_charge;

    void init_ewald_sum(YAML::Node& doc);
    double potential_ewald_sum();
    double compute_ewald_real();
    double compute_ewald_reciprocal();
    double compute_ewald_self();

    struct Tag_potential_ewald_real {};
    struct Tag_potential_ewald_reciprocal {};
    struct Tag_potential_ewald_self {};
    KOKKOS_FUNCTION void operator() (Tag_potential_ewald_real, const member_type& teamMember, double& V) const;
    KOKKOS_FUNCTION void operator() (Tag_potential_ewald_reciprocal, const member_type& teamMember, double& V) const;
    KOKKOS_FUNCTION void operator() (Tag_potential_ewald_self, const member_type& teamMember, double& V) const;


    // Bonds/angles
    void read_bonds_angles(const std::string& filename);
    Kokkos::View<Bond*> bonds;
    Kokkos::View<Bond*>::HostMirror h_bonds;
    Kokkos::View<BondType*> bondTypes;
    Kokkos::View<BondType*>::HostMirror h_bondTypes;
    Kokkos::View<AngleType*> angleTypes;
    Kokkos::View<AngleType*>::HostMirror h_angleTypes;
    Kokkos::View<Angle*> angles;
    Kokkos::View<Angle*>::HostMirror h_angles;

    double potential_bonds_angles();

    void build_bondless_verlet_list() override;
    struct Tag_verlet_remove_bonds {};
    KOKKOS_FUNCTION void operator() (Tag_verlet_remove_bonds, const member_type& teamMember) const;

    double potential_bonds();
    struct Tag_potential_bonds {};
    KOKKOS_FUNCTION void operator() (Tag_potential_bonds, const member_type& team_member, double& V) const;

    double potential_angles();
    struct Tag_potential_angles {};
    KOKKOS_FUNCTION void operator() (Tag_potential_angles, const member_type& team_member, double& V) const;
    
    void compute_force_bonds_angles();
    void compute_force_bonds();
    struct Tag_force_bonds {};
    KOKKOS_FUNCTION void operator() (Tag_force_bonds, const member_type& team_member) const;

    void compute_force_angles();
    struct Tag_force_angles {};
    KOKKOS_FUNCTION void operator() (const int i)const;


    // Destructor
    ~particles_instance() {};
};


// we need a ruduction of 3 double array
template< class ScalarType, int N >
struct array_type {
    ScalarType the_array[N];

    KOKKOS_INLINE_FUNCTION   // Default constructor - Initialize to 0's
        array_type() {
        for (int i = 0; i < N; i++) { the_array[i] = 0; }
    }
    KOKKOS_INLINE_FUNCTION   // Copy Constructor
        array_type(const array_type& rhs) {
        for (int i = 0; i < N; i++) {
            the_array[i] = rhs.the_array[i];
        }
    }
    KOKKOS_INLINE_FUNCTION   // add operator
        array_type& operator += (const array_type& src) {
        for (int i = 0; i < N; i++) {
            the_array[i] += src.the_array[i];
        }
        return *this;
    }
};
typedef array_type<double, dim_space> space_vector;  // used to simplify code below
namespace Kokkos { //reduction identity must be defined in Kokkos namespace
    template<>
    struct reduction_identity< space_vector > {
        KOKKOS_FORCEINLINE_FUNCTION static space_vector sum() {
            return space_vector();
        }
    };
}

#endif