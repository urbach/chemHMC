#ifndef particles_H
#define particles_H

#include <functional>
#include <vector>
#include <atom.hpp>
#include <iostream>
#include <iomanip>
#include "yaml-cpp/yaml.h"
#include "global.hpp"
#include "../modules/potentials/molecules/bonds.hpp"
#include "Parameters.hpp"
#include "../modules/neighbor_list/Neighbor_list.hpp"
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
    double inverse_L[dim_space];    // 1/L
    double inverse_halved_L[dim_space]; // 2/L
    double beta;
    double sbeta;// sqrt(beta)
    double cutoff;
    double cutoff_squared;
    std::string name_xyz;
    bool neighbor_list_used = false;
    int number_of_molecules = 0;

    type_x  x;      ///< Kokkos view containing the positions
    type_p  p;      ///< Kokkos view containing the momenta
    type_f  f;      ///< Kokkos view containing the forces
    type_id id;     ///< Kokkos view containing the type_ids 
    // the host mirror of x is used to restore the position before the MD in case of a rejection
    type_x::HostMirror h_x;     ///< Host mirror of x (positions).
    type_p::HostMirror h_p;     ///< Host_mirror of p (momenta)
    type_f::HostMirror h_f;     ///< Host_mirror of f (forces)
    type_id::HostMirror h_id;   ///< host mirror of id

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
    Kokkos::View<int*> mol_id; ///< mapping from atom id to molecule id
    Kokkos::View<int*> h_mol_id; ///< host mirror of mol_id


    double T;   ///< temperature

    // file interaction
    void print_xyz(params_class params, int traj, double K, double V);
    void print_force(params_class params, int traj);
    void print_xyz_and_momenta(params_class params, int traj, double K, double V);

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

    // force calculation
    struct force {};
    
    std::function<void()>  force_strategy;
    
    void compute_force() {
        force_strategy();
    };

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