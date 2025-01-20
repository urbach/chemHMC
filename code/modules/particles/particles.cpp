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
    f = type_f("f", N); // forces
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
    //compute_force();
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

double dot_product(const int N, const type_p& a, const type_p& b) {
    double result = 0.0;
    Kokkos::parallel_reduce("DotProduct", Kokkos::RangePolicy<>(0, N), KOKKOS_LAMBDA(const int i, double& thread_sum) {
        thread_sum += a(i, 0) * b(i, 0) + a(i, 1) * b(i, 1) + a(i, 2) * b(i, 2);
    }, result);
    return result;
}

void particles_instance::minimize_energy(YAML::Node& doc) {
    std::string minimization_algorithm = check_and_assign_value<std::string>(doc["minimization"], "algorithm");
    double V;
    if (minimization_algorithm == "gradient_descent") {
        V = gradient_descent_minimzation(doc);
    }   else if (minimization_algorithm == "conjugate_gradient") {
        V = conjugate_gradient_minimzation(doc);
    }

    // Transfer the optimized positions back to the host
    Kokkos::deep_copy(h_x, x);

    // Save the optimized geometry if requested
    if (doc["minimization"]["save_geometry"]) {
        save_optimized_geometry(V);
    }
}

double particles_instance::conjugate_gradient_minimzation(YAML::Node& doc) {
    // Extract configuration values only once
    int max_iter = doc["minimization"]["max_iter"] ? check_and_assign_value<int>(doc["minimization"], "max_iter") : 1000;
    double tolerance = doc["minimization"]["tolerance"] ? check_and_assign_value<double>(doc["minimization"], "tolerance") : 1e-6;
    double dt = check_and_assign_value<double>(doc["integrator"], "dt");

    // Precompute invariant data
    if (algorithm == "verlet_list") {
        //build_verlet_list();
    } else if (algorithm == "bonds_angles") {
        //build_bondless_verlet_list();
    }

    // Initialize variables
    double V = compute_potential();
    double V_new;

    // Allocate vectors for CG
    type_p g("gradient", N);      // Gradient vector (forces)
    type_p h("direction", N);     // Search direction vector
    type_p g_old("gradient_old", N);  // Old gradient vector
    type_p temp("temp_vector", N);    // Temporary vector for calculations

    // Initialize the gradient and search direction
    compute_force();
    Kokkos::deep_copy(g, f);  // f stores the forces (negative gradient of the potential)
    Kokkos::deep_copy(h, g);  // Initial direction is the same as the gradient

    double g_norm2 = dot_product(N, g, g); // Compute norm squared of the gradient

    for (int iter = 0; iter < max_iter; iter++) {
        // Manually update momenta: p = p + dt * h
        Kokkos::deep_copy(f, h);
        update_momenta(dt);
        update_positions(dt);
        // Compute new potential
        V_new = compute_potential();
        // Check for convergence
        if (fabs(V_new - V) < tolerance) break;
        V = V_new;
        // Compute new gradient
        Kokkos::deep_copy(g_old, g); // Store old gradient
        compute_force();             // Update forces
        Kokkos::deep_copy(g, f);     // Update gradient vector

        double g_new_norm2 = dot_product(N, g, g); // Compute norm squared of new gradient
        // Compute beta (Polak-Ribiere method)
        double beta = (g_new_norm2 - dot_product(N, g, g_old)) / g_norm2;
        g_norm2 = g_new_norm2; // Update g_norm2 for next iteration
        // Update search direction: h = g + beta * h
        Kokkos::parallel_for("UpdateDirection", Kokkos::RangePolicy<>(0, N), KOKKOS_LAMBDA(const int i) {
            for (int d = 0; d < dim_space; d++) {
                h(i, d) = g(i, d) + beta * h(i, d);
            }
        });
    }

    return V;
}

double particles_instance::gradient_descent_minimzation(YAML::Node& doc) {

    // get config
    int max_iter = 0;
    if (doc["minimization"]["max_iter"]) {
        max_iter = check_and_assign_value<int>(doc["minimization"],"max_iter");
    } else {
        max_iter = 1000; //default value for maximum iterations
    }
    double tolerance = 0.0;
    if (doc["minimization"]["tolerance"]) {
        tolerance = check_and_assign_value<double>(doc["minimization"],"tolerance");
    } else {
        tolerance = 1e-6; //default value for energy tolerance
    }
    double dt = check_and_assign_value<double>(doc["integrator"],"dt");
    // get initial potential energy
    //if (algorithm == "verlet_list") build_verlet_list();
    //if (algorithm == "bonds_angles") build_bondless_verlet_list();
    double V = compute_potential();
    double V_new;
    // Start energy minimization
    for (int i = 0; i < max_iter; i++) {
        Kokkos::deep_copy(p,0);
        compute_force();
        update_momenta(dt);
        update_positions(dt);
        V_new = compute_potential();
        if(abs(V_new - V) < tolerance) {
            V = V_new;
            printf("MINIMIZATION CONVERGED \n");
            break;
        }
        V = V_new;
    }

    return V;
}

void particles_instance::save_optimized_geometry(double V) const {
    FILE* opt_file = fopen("optimized_structure.xyz", "ab");
    if (opt_file) {
        fprintf(opt_file, "     %d\n", N);
        fprintf(opt_file, "optimized geometry. V=%f\n", V);
        for (int i = 0; i < N; ++i) {
            fprintf(opt_file, "%-8s  %20.12g  %20.12g  %20.12g\n",
                    label_xyz[i].c_str(), h_x(i, 0), h_x(i, 1), h_x(i, 2));
        }
        fclose(opt_file);
    }
}