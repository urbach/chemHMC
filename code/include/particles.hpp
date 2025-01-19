#ifndef particles_H
#define particles_H

#include <functional>
#include <vector>
#include <atom.hpp>
#include <iostream>
#include <iomanip>
#include "yaml-cpp/yaml.h"
#include "global.hpp"
#include "bonds.hpp"
#include "Parameters.hpp"
#include "Neighbor_list.hpp"
#include <Kokkos_Core.hpp>

class particles_instance {

public:
    typedef Kokkos::TeamPolicy<>::member_type  member_type;
    
    Neighbor_list* neighbor_list = nullptr;
    std::shared_ptr<Bonds> bonds_ptr = nullptr;

    int N;
    int seed;
    double coeff_p;
    double L[dim_space];
    double inverse_L[dim_space];
    double inverse_halved_L[dim_space];
    double beta;
    double sbeta;// sqrt(beta)
    double cutoff;
    double cutoff_squared;
    std::string name_xyz;

    type_x  x;      ///< Kokkos view containing the positions
    type_p  p;      ///< Kokkos view containing the momenta
    type_f  f;      ///< Kokkos view containing the forces
    type_id id;     ///< Kokkos view containing the type_ids 
    // the host mirror of x is used to restore the position before the MD in case of a rejection
    type_x::HostMirror h_x;     ///< Host mirror of x (positions).
    type_p::HostMirror h_p;     ///< Host_mirror of p (momenta)

    std::string algorithm;
    std::vector<std::string> label_xyz;

    // rng
    RandPoolType rand_pool;

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

    // verlet_list related stuff
    Kokkos::View<int*> neighbour_count;
    Kokkos::View<int*>::HostMirror h_neighbour_count;
    Kokkos::View<int**> verlet_list;
    Kokkos::View<int**>::HostMirror h_verlet_list;

    // file interaction
    void print_xyz(params_class params, int traj, double K, double V);

    // initialization related stuff
    struct check_in_volume {};

    void InitX();
    double get_beta() { return beta; };
    void assign_algorithm(YAML::Node& doc);
    void compute_coeff_position();
    void compute_coeff_momenta();

    KOKKOS_FUNCTION void operator() (check_in_volume, const int i) const;
    

    // integrator related stuff
    struct hbTag {};

    void hb();
    void update_positions(const double dt_);
    void update_momenta(const double dt_);

    KOKKOS_FUNCTION void operator() (hbTag, const int i) const;
    
    // kinetic energy calculation
    struct kinetic {};

    double compute_kinetic_E();

    KOKKOS_FUNCTION void operator() (kinetic, const int& i, double& sum) const;

    std::function<double()>  potential_strategy;
    
    double compute_potential() {
        return potential_strategy();
    };

    // pot E minimization
    void minimize_energy(YAML::Node& doc);
    double gradient_descent_minimzation(YAML::Node& doc);
    double conjugate_gradient_minimzation(YAML::Node& doc);
    void save_optimized_geometry(double V) const;

    // force calculation
    struct force {};
    
    std::function<void()>  force_strategy;
    
    void compute_force() {
        force_strategy();
    };

    // Ewald sum
    double ewald_alpha; // width of the gaussians
    double sqrt_ewald_alpha;
    double r_c; // real-space cutoff for the ewald sum
    double r_c2;
    double ewald_accuracy; // rms accuracy of the ewald sum
    int k_max;
    double V_self = 0.0; // self interaction energy in the ewald sum is precomputed and stored.
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

    void compute_force_ewald();
    void compute_ewald_real_forces();
    void compute_ewald_reciprocal_forces();

    struct Tag_force_ewald_real {};
    struct Tag_force_ewald_reciprocal {};

    KOKKOS_FUNCTION void operator()(Tag_force_ewald_real, const member_type& teamMember) const;
    KOKKOS_FUNCTION void operator()(Tag_force_ewald_reciprocal, const member_type& teamMember) const;

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
    Kokkos::View<DihedralType*> dihedralTypes;
    Kokkos::View<DihedralType*>::HostMirror h_dihedralTypes;
    Kokkos::View<Dihedral*> dihedrals;
    Kokkos::View<Dihedral*>::HostMirror h_dihedrals;

    double potential_bonds_angles();

    void build_bondless_verlet_list();
    struct Tag_verlet_remove_bonds {};
    KOKKOS_FUNCTION void operator() (Tag_verlet_remove_bonds, const member_type& teamMember) const;

    double potential_bonds();
    struct Tag_potential_bonds {};
    KOKKOS_FUNCTION void operator() (Tag_potential_bonds, const member_type& team_member, double& V) const;

    double potential_angles();
    struct Tag_potential_angles {};
    KOKKOS_FUNCTION void operator() (Tag_potential_angles, const member_type& team_member, double& V) const;
    
    double potential_dihedrals();
    struct Tag_potential_dihedrals {};
    KOKKOS_FUNCTION void operator() (Tag_potential_dihedrals, const member_type& team_member, double& V) const;

    void compute_force_bonds_angles();
    void compute_force_bonds();
    struct Tag_force_bonds {};
    KOKKOS_FUNCTION void operator() (Tag_force_bonds, const member_type& team_member) const;

    void compute_force_angles();
    struct Tag_force_angles {};
    KOKKOS_FUNCTION void operator() (const int i)const;

    void compute_force_dihedrals();
    struct Tag_force_dihedrals {};
    KOKKOS_FUNCTION void operator() (Tag_force_dihedrals, const member_type& team_member) const;

    

    // OPLS
    double potential_opls();
    void compute_force_opls();

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