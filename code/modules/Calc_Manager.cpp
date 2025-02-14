#include "Calc_Manager.hpp"
#include <iostream>
#include "Input_reader.hpp"
#include "global.hpp"

void Calc_Manager::addCalc(std::shared_ptr<Calc> calc) {
    calc_list.push_back(calc);
}

void Calc_Manager::initialize() {
    for (const auto& calc : calc_list) {
        if (calc) { // Check if the pointer is valid
            calc->init( *particles );
        }
    }
}

void Calc_Manager::compute_force() {
    // Set forces to 0
    Kokkos::deep_copy(particles->f, 0.0);
    // Loop over different potentials in calc_list
    for (const auto& calc : calc_list) {
            calc->force( *particles, particles->f );
    }
}

double Calc_Manager::compute_potential() {
    double result = 0;
    // Loop over different potentials in calc_list
    for (const auto& calc : calc_list) {
            result += calc->potential( *particles );
    }
    return result;
}

void Calc_Manager::set_particles(std::shared_ptr<particles_instance> particles_in) {
    particles = particles_in;
}

void Calc_Manager::print_timings() {
    for (const auto& calc : calc_list) {
        calc->print_timings();
    }
}

//
// MINIMIZATION
//

double dot_product(const int N, const type_p& a, const type_p& b) {
    double result = 0.0;
    Kokkos::parallel_reduce("DotProduct", Kokkos::RangePolicy<>(0, N), KOKKOS_LAMBDA(const int i, double& thread_sum) {
        thread_sum += a(i, 0) * b(i, 0) + a(i, 1) * b(i, 1) + a(i, 2) * b(i, 2);
    }, result);
    return result;
}

void Calc_Manager::minimize_energy(YAML::Node& doc) {
    std::string minimization_algorithm = check_and_assign_value<std::string>(doc["minimization"], "algorithm");
    double V;
    if (minimization_algorithm == "gradient_descent") {
        V = gradient_descent_minimzation(doc);
    }   else if (minimization_algorithm == "conjugate_gradient") {
        V = conjugate_gradient_minimzation(doc);
    }

    // Transfer the optimized positions back to the host
    Kokkos::deep_copy(particles->h_x, particles->x);

    // Save the optimized geometry if requested
    if (doc["minimization"]["save_geometry"]) {
        save_optimized_geometry(V);
    }
}

double Calc_Manager::conjugate_gradient_minimzation(YAML::Node& doc) {
    // Extract configuration values only once
    int max_iter = doc["minimization"]["max_iter"] ? check_and_assign_value<int>(doc["minimization"], "max_iter") : 1000;
    double tolerance = doc["minimization"]["tolerance"] ? check_and_assign_value<double>(doc["minimization"], "tolerance") : 1e-6;
    double dt = check_and_assign_value<double>(doc["integrator"], "dt");

    // Initialize variables
    double V = compute_potential();
    double V_new;

    // Allocate vectors for CG
    type_p g("gradient", particles->N);      // Gradient vector
    type_p h("direction", particles->N);     // Search direction vector
    type_p g_old("gradient_old", particles->N);  // Old gradient vector
    type_p temp("temp_vector", particles->N);    // Temporary vector for calculations

    // Initialize the gradient and search direction
    compute_force();
    Kokkos::deep_copy(g, particles->f);  // f stores the forces
    Kokkos::deep_copy(h, g);  // Initial direction is the same as the gradient

    double g_norm2 = dot_product(particles->N, g, g); // Compute norm squared of the gradient

    for (int iter = 0; iter < max_iter; iter++) {
        // Manually update momenta: p = p + dt * h
        Kokkos::deep_copy(particles->f, h);
        particles->update_momenta(dt);
        particles->update_positions(dt);
        // Compute new potential
        V_new = compute_potential();
        // Check for convergence
        if (fabs(V_new - V) < tolerance) break;
        V = V_new;
        // Compute new gradient
        Kokkos::deep_copy(g_old, g); // Store old gradient
        compute_force();             // Update forces
        Kokkos::deep_copy(g, particles->f);     // Update gradient vector

        double g_new_norm2 = dot_product(particles->N, g, g); // Compute norm squared of new gradient
        // Compute beta (Polak-Ribiere method)
        double beta = (g_new_norm2 - dot_product(particles->N, g, g_old)) / g_norm2;
        g_norm2 = g_new_norm2; // Update g_norm2 for next iteration
        // Update search direction: h = g + beta * h
        Kokkos::parallel_for("UpdateDirection", Kokkos::RangePolicy<>(0, particles->N), KOKKOS_LAMBDA(const int i) {
            for (int d = 0; d < dim_space; d++) {
                h(i, d) = g(i, d) + beta * h(i, d);
            }
        });
    }

    return V;
}

double Calc_Manager::gradient_descent_minimzation(YAML::Node& doc) {
    double tokcal = 1.0/kcaltointernal;
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
    double V = compute_potential();
    double V_new;
    // Start energy minimization
    for (int i = 0; i < max_iter; i++) {
        Kokkos::deep_copy(particles->p,0.0);
        compute_force();
        Kokkos::deep_copy(particles->h_f,particles->f);
        particles->update_momenta(dt);
        Kokkos::deep_copy(particles->h_p,particles->p);
        particles->update_positions(dt);
        V_new = compute_potential();
        if((std::fabs(V_new - V)*tokcal) < tolerance) {
            V = V_new;
            printf("MINIMIZATION CONVERGED \n");
            break;
        }
        V = V_new;
    }

    return V*tokcal;
}

void Calc_Manager::save_optimized_geometry(double V) const {
    FILE* opt_file = fopen("optimized_structure.xyz", "ab");
    if (opt_file) {
        fprintf(opt_file, "     %d\n", particles->N);
        fprintf(opt_file, "optimized geometry. V=%f\n", V);
        for (int i = 0; i < particles->N; ++i) {
            fprintf(opt_file, "%-8s  %20.12g  %20.12g  %20.12g\n",
                particles->label_xyz[i].c_str(), particles->h_x(i, 0), particles->h_x(i, 1), particles->h_x(i, 2));
        }
        fclose(opt_file);
    }
}