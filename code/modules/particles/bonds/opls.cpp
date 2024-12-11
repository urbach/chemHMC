#include "global.hpp"
#include "atom.hpp"
#include "particles.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

double particles_instance::potential_opls() {
    double result = potential_verlet_list();
    //result += potential_ewald_sum();
    result += potential_bonds_angles();
    return result;
}

void particles_instance::compute_force_opls() {
    // reset force
    Kokkos::deep_copy(f,0);

    //compute bonded forces
    compute_force_bonds();
    compute_force_angles();
    compute_force_dihedrals();

    //compute non-bonded forces
    typedef Kokkos::TeamPolicy<Tag_force_verlet> team_policy;
    Kokkos::parallel_for("compute_force_verlet_list", team_policy(N, Kokkos::AUTO), *this);
    compute_ewald_real_forces();
    compute_ewald_reciprocal_forces();
}