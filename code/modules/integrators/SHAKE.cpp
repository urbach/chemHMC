#include "SHAKE.hpp"

VELOCITY_VERLET_SHAKE::VELOCITY_VERLET_SHAKE(YAML::Node doc, params_class params): integrator_type(doc, params) {}

void VELOCITY_VERLET_SHAKE::integrate() {
    calc_manager->compute_force();
    Kokkos::fence();
    for (size_t i = 0; i < steps; i++) {
        particles->update_momenta(dt / 2.);
        particles->update_positions(dt);

        // Apply SHAKE to correct positions
        apply_SHAKE();
        calc_manager->compute_force();
        Kokkos::fence();
        particles->update_momenta(dt / 2.);

        // Apply RATTLE to correct velocities
        apply_RATTLE();
    }
}

// SHAKE: Corrects bond constraints on positions
void VELOCITY_VERLET_SHAKE::apply_SHAKE() {
    const int max_iter = 100;  // Max iterations
    const double tolerance = 1e-4;  // Convergence criterion

    auto& x = particles->x;
    auto& L = particles->L;
    auto& id = particles->id;
    auto& inverse_halved_L = particles->inverse_halved_L;
    auto& bonds = particles->bonds_ptr->constrained_bonds;
    auto& bondTypes = particles->bonds_ptr->bondTypes;
    auto& coeff_x = particles->coeff_x; // 1/mass lookup table



    for (int iter = 0; iter < max_iter; iter++) {
        double max_error = 0.0;
        //printf("\n SHAKE ITERATION %d \n\n", iter+1);
        Kokkos::parallel_reduce(
        "SHAKE",
        Kokkos::RangePolicy<>(0, bonds.extent(0)),
        KOKKOS_LAMBDA(const int i, double& local_max_error) {

            int atom1 = bonds(i).atom1;
            int atom2 = bonds(i).atom2;
            int type1 = id(atom1);
            int type2 = id(atom2);
            int bondtype = bonds(i).type;
            double r0 = bondTypes(bondtype).r0;

            double r[3];
            double r2 = 0.0;

            // Calculate displacement vector between atoms
            for (int dim = 0; dim < 3; dim++) {
                r[dim] = x(atom1, dim) - x(atom2, dim);
                r[dim] -= round(r[dim] * inverse_halved_L[dim]) * L[dim];
                r2 += r[dim] * r[dim];
            }

            // Calculate deviation from equilibrium bond length
            double r_norm = sqrt(r2 + 1e-12);
            double error = r_norm - r0;
            local_max_error = fmax(local_max_error, fabs(error));

            // If deviation exceeds tolerance, apply shake correction
            if (fabs(error) > tolerance) {
                // Compute lagrange multiplier
                double sum_inv_mass = coeff_x(type1) + coeff_x(type2);
                double lambda = (0.5 * (error / r_norm)) / sum_inv_mass;

                for (int dim = 0; dim < 3; dim++) {
                    // Scale correction factor by mass and apply it
                    Kokkos::atomic_add(&x(atom1, dim), -(coeff_x(atom1)*lambda) * r[dim]);
                    Kokkos::atomic_add(&x(atom2, dim), (coeff_x(atom2)*lambda) * r[dim]);
                }
            }
        },
        max_error);

        // If constraints are satisfied, exit early
        if (max_error < tolerance) break;
    }
}

// RATTLE: Corrects velocities to satisfy constraints
void VELOCITY_VERLET_SHAKE::apply_RATTLE() {
    const int max_iter = 100;
    const double tolerance = 1e-4;

    auto& p = particles->p;  // Momenta
    auto& x = particles->x;  // Positions
    auto& L = particles->L;
    auto& id = particles->id;
    auto& inverse_halved_L = particles->inverse_halved_L;
    auto& bonds = particles->bonds_ptr->constrained_bonds;
    auto& coeff_x = particles->coeff_x;  // 1/mass lookup table

    for (int iter = 0; iter < max_iter; iter++) {
        double max_error = 0.0;

        Kokkos::parallel_reduce(
        "RATTLE",
        Kokkos::RangePolicy<>(0, bonds.extent(0)),
        KOKKOS_LAMBDA(const int i, double& local_max_error) {

            int atom1 = bonds(i).atom1;
            int atom2 = bonds(i).atom2;
            int type1 = id(atom1);
            int type2 = id(atom2);

            double m1_inv = coeff_x(type1);
            double m2_inv = coeff_x(type2);

            // Compute relative position vector with periodic boundary conditions
            double r[3], r2 = 0.0;
            for (int dim = 0; dim < 3; dim++) {
                r[dim] = x(atom1, dim) - x(atom2, dim);
                r[dim] -= round(r[dim] * inverse_halved_L[dim]) * L[dim]; 
                r2 += r[dim] * r[dim];
            }
            double r_norm = sqrt(r2 + 1e-12);  // Avoid divide-by-zero

            // Compute relative momentum
            double pij[3], dot_product = 0.0;
            for (int dim = 0; dim < 3; dim++) {
                pij[dim] = p(atom1, dim) * m1_inv - p(atom2, dim) * m2_inv;
                dot_product += pij[dim] * r[dim];
            }

            // Compute velocity constraint error
            double error = dot_product / r_norm;
            local_max_error = fmax(local_max_error, fabs(error));

            if (fabs(error) > tolerance) {
                double sum_inv_mass = m1_inv + m2_inv;
                double correction_factor = error / (r_norm * sum_inv_mass);

                for (int dim = 0; dim < 3; dim++) {
                    double correction = correction_factor * r[dim];
                    Kokkos::atomic_add(&p(atom1, dim), -correction);
                    Kokkos::atomic_add(&p(atom2, dim), correction);
                }
            }
        },
        max_error);

        if (max_error < tolerance) break;
    }
}