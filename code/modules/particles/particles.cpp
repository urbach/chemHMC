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
#include "Input_reader.hpp"
#include "Parameters.hpp"
#include "particles.hpp"
#include "Neighbor_list.hpp"

void particles_instance::InitX() {
    // Create all general kokkos views that are needed
    x = type_x("x", N); // particle positions
    h_x = Kokkos::create_mirror(x);
    p = type_p("p", N); // particle momenta
    h_p = Kokkos::create_mirror(p);
    f = type_f("f", N); // forces
    h_f = Kokkos::create_mirror(f);
    id = type_id("id", N); // particle type id
    h_id = Kokkos::create_mirror(id);
}

void particles_instance::compute_coeff_momenta() {
    coeff_p = 1.0;//beta;
}

void particles_instance::compute_coeff_position() {

    // inititalize device and host views
    coeff_x = Kokkos::View<double*>("coeff_x",h_atom_type_list.extent(0));
    h_coeff_x = Kokkos::create_mirror_view(coeff_x);
    //Since we have different particles  we need to compute one coefficient for each type
    for(int i = 0;i < h_atom_type_list.extent(0);i++) {
        h_coeff_x[i] = 1.0 / (h_atom_type_list[i].mass);//beta / (h_atom_type_list[i].mass);
    }
    //copy to device
    Kokkos::deep_copy(coeff_x, h_coeff_x);
}

double particles_instance::compute_kinetic_E() {
    double K = 0.0;

    // Capture all needed members explicitly
    auto& p = this->p;
    auto& id = this->id;
    auto& atom_type_list = this->atom_type_list;

    // Use parallel_reduce with the tag
    Kokkos::parallel_reduce(
        "identical-particles-LJ-kinetic-E",
        Kokkos::RangePolicy<kinetic>(0, N),
        KOKKOS_LAMBDA(const kinetic&, const int i, double& sum) {
            double mass = atom_type_list[id(i)].mass;
            double kinetic_energy = (p(i, 0) * p(i, 0) + p(i, 1) * p(i, 1) + p(i, 2) * p(i, 2)) / (2 * mass);
            sum += kinetic_energy;
        },
        K);

    return K;
}

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
            x(i, dir) += dt * c[id[i]] * p(i, dir);
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

void particles_instance::print_xyz(params_class params, int traj, double K, double V) {
    fprintf(params.fileout, "     %d\n", N);
    fprintf(params.fileout, "trajectory= %d  kinetic_energy= %.12g  potential= %.12g\n", traj, K/kcaltointernal, V/kcaltointernal);
    for (int i = 0; i < N; i++)
        fprintf(params.fileout, "%s  %-20.12g %-20.12g %-20.12g\n", label_xyz[i].c_str(), h_x(i, 0), h_x(i, 1), h_x(i, 2));
}

void particles_instance::print_force(params_class params, int traj) {
    fprintf(params.fileout, "     %d\n", N);
    fprintf(params.fileout, "trajectory= %d  FORCES\n", traj);
    for (int i = 0; i < N; i++)
        fprintf(params.fileout, "%s  %-20.12g %-20.12g %-20.12g\n", label_xyz[i].c_str(), h_f(i, 0)/internalforcetolammpsreal, h_f(i, 1)/internalforcetolammpsreal, h_f(i, 2)/internalforcetolammpsreal);
}

void particles_instance::hb() {
    Kokkos::parallel_for("hb_momenta", Kokkos::RangePolicy<hbTag>(0, N), *this);
}

KOKKOS_FUNCTION
void particles_instance::operator() (hbTag, const int i) const {
    gen_type rgen = rand_pool.get_state(i);
    // we need to divide by sqrt(2) in order to have exp(-p^2)
    // normal() produced distribution exp(-p^2/2)
    p(i, 0) = rgen.normal() * Kokkos::sqrt(atom_type_list[id[i]].mass / beta);
    p(i, 1) = rgen.normal() * Kokkos::sqrt(atom_type_list[id[i]].mass / beta);
    p(i, 2) = rgen.normal() * Kokkos::sqrt(atom_type_list[id[i]].mass / beta);
    rand_pool.free_state(rgen);
}