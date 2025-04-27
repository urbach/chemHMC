#include "bonds.hpp"
#include "particles.hpp"
#include "global.hpp"

void Bonds::init(const particles_instance& particles) {}

void Bonds::build_constrained_bond_list(std::vector<int> constrained_bond_type_indices) {
    // Go through  the bonds list to count the number of bonds that need to be constrained
    size_t number_of_constrained_bonds = 0;
    for (size_t i = 0; i < h_bonds.extent(0); i++) {
        for (size_t j = 0; j < constrained_bond_type_indices.size(); j++) {
            if (h_bonds(i).type == constrained_bond_type_indices[j]-1) {
                number_of_constrained_bonds++;
                break;
            }
        }
    }
    // allocate views
    constrained_bonds = Kokkos::View<Bond*>("contrained_bond_list", number_of_constrained_bonds);
    h_constrained_bonds = Kokkos::create_mirror_view(constrained_bonds);
    size_t number_of_unconstrained_bonds = h_bonds.extent(0)-number_of_constrained_bonds;
    unconstrained_bonds = Kokkos::View<Bond*>("uncontrained_bond_list", number_of_unconstrained_bonds);
    h_unconstrained_bonds = Kokkos::create_mirror_view(unconstrained_bonds);
    // Now that we have allocated space we can populate the lists
    size_t const_idx = 0;
    size_t unconst_idx = 0;
    for (size_t i = 0; i < h_bonds.extent(0); i++) {
        bool is_constrained = false;
        for (size_t j = 0; j < constrained_bond_type_indices.size(); j++) {
            if (h_bonds(i).type == constrained_bond_type_indices[j]-1) {
                is_constrained = true;
                break;  // Avoid adding the same bond multiple times
            }
        }
        if (is_constrained) {
            h_constrained_bonds(const_idx) = h_bonds(i);
            const_idx++;
        } else {
            h_unconstrained_bonds(unconst_idx) = h_bonds(i);
            unconst_idx++;
        }
    }
    Kokkos::deep_copy(constrained_bonds,h_constrained_bonds);
    Kokkos::deep_copy(unconstrained_bonds,h_unconstrained_bonds);
}

double Bonds::potential(const particles_instance& particles) {
    Kokkos::Timer bonds_timer;
    double result = 0.0;
    result += potential_bonds(particles);
    result += potential_angles(particles);
    result += potential_dihedrals(particles);
    time_potential += bonds_timer.seconds();
    return result;
}

void Bonds::force(const particles_instance& particles, type_f& f) {
    Kokkos::Timer bonds_timer;
    force_bonds(particles,f);
    force_angles(particles,f);
    force_dihedrals(particles,f);
    time_force += bonds_timer.seconds();
}

double Bonds::potential_bonds(const particles_instance& particles) {
    double result = 0.0;

    // Capture all needed members of particles_instance
    auto& x = particles.x;
    auto& L = particles.L;
    auto& inverse_halved_L = particles.inverse_halved_L;
    auto& bonds = this->unconstrained_bonds;
    auto& bondTypes = this->bondTypes;

    // Outer parallel_reduce
    Kokkos::parallel_reduce(
        "bond-potential",
        Kokkos::RangePolicy<>(0, bonds.extent(0)),
        KOKKOS_LAMBDA(const int i, double& V) {

            int atom1 = bonds(i).atom1;
            int atom2 = bonds(i).atom2;
            int type = bonds(i).type;
            double k = bondTypes(type).k;
            double r0 = bondTypes(type).r0;
            double r = (x(atom1,0) - x(atom2,0));
            r -= int(r * inverse_halved_L[0]) * L[0];
            double r2 = r*r;
            r = (x(atom1,1) - x(atom2,1));
            r -= int(r * inverse_halved_L[1]) * L[1];
            r2 += r*r;
            r = (x(atom1,2) - x(atom2,2));
            r -= int(r * inverse_halved_L[2]) * L[2];
            r2 += r*r;
            r = sqrt(r2);

            double dr = r-r0;
            double potential = k*dr*dr; // The usual factor of 0.5 is already part of k

            V += potential;
        },
    result);

    return result;
}

double Bonds::potential_angles(const particles_instance& particles) {
    double result = 0.0;

    // Capture all needed members of particles_instance
    auto& x = particles.x;
    auto& L = particles.L;
    auto& inverse_halved_L = particles.inverse_halved_L;
    auto& angles = this->angles;
    auto& angleTypes = this->angleTypes;

    // Outer parallel_reduce
    Kokkos::parallel_reduce(
        "angles-potential",
        Kokkos::RangePolicy<>(0, angles.extent(0)),
        KOKKOS_LAMBDA(int i, double &V) {

            int atom1 = angles(i).atom1;
            int atom2 = angles(i).atom2;
            int atom3 = angles(i).atom3;
            int type = angles(i).type;
            double k = angleTypes(type).k;
            double theta0 = angleTypes(type).theta0;

            //Kokkos::printf("atom1: %d atom2: %d atom3: %d type: %d k: %f theta0: %f\n", atom1,atom2,atom3,type,k,theta0);
            // Vector from atom2 to atom1
            double v1x = x(atom1,0) - x(atom2,0);
            v1x -= int(v1x * inverse_halved_L[0]) * L[0];
            double v1y = x(atom1,1) - x(atom2,1);
            v1y -= int(v1y * inverse_halved_L[1]) * L[1];
            double v1z = x(atom1,2) - x(atom2,2);
            v1z -= int(v1z * inverse_halved_L[2]) * L[2];

            // Vector from atom2 to atom3
            double v2x = x(atom3,0) - x(atom2,0);
            v2x -= int(v2x * inverse_halved_L[0]) * L[0];
            double v2y = x(atom3,1) - x(atom2,1);
            v2y -= int(v2y * inverse_halved_L[1]) * L[1];
            double v2z = x(atom3,2) - x(atom2,2);
            v2z -= int(v2z * inverse_halved_L[2]) * L[2];

            // Dot product of v1 and v2
            double dot_product = v1x * v2x + v1y * v2y + v1z * v2z;

            // Magnitudes of vectors
            double v1_mag = sqrt(v1x*v1x + v1y*v1y + v1z*v1z);
            double v2_mag = sqrt(v2x*v2x + v2y*v2y + v2z*v2z);

            // Cosine of the angle
            double cos_theta = dot_product / (v1_mag * v2_mag);

            // Compute the angle
            double theta = acos(cos_theta);

            // Calculate the deviation from the equilibrium angle
            double dtheta = theta - theta0;

            // Compute the angle potential
            double potential = k * dtheta * dtheta; // The usual factor of 0.5 is already part of k

            V += potential;
        },
    result);
    return result;
}

double Bonds::potential_dihedrals(const particles_instance& particles) {
    double result = 0.0;

    // Capture all needed members of particles_instance
    auto& x = particles.x;
    auto& L = particles.L;
    auto& inverse_L = particles.inverse_L;
    auto& dihedrals = this->dihedrals;
    auto& dihedralTypes = this->dihedralTypes;

    // Outer parallel_reduce
    Kokkos::parallel_reduce(
        "dihedrals-potential",
        Kokkos::RangePolicy<>(0, dihedrals.extent(0)),
        KOKKOS_LAMBDA(int i, double &V) {

            int atom1 = dihedrals(i).atom1;
            int atom2 = dihedrals(i).atom2;
            int atom3 = dihedrals(i).atom3;
            int atom4 = dihedrals(i).atom4;
            int type = dihedrals(i).type;

            double k1 = dihedralTypes(type).k1;
            double k2 = dihedralTypes(type).k2;
            double k3 = dihedralTypes(type).k3;
            double k4 = dihedralTypes(type).k4;

            // Positions and displacement vectors with periodic boundary conditions
            double x1[3], x2[3], x3[3], x4[3];
            double r12[3], r23[3], r34[3];

            for (int d = 0; d < 3; ++d) {
                x1[d] = x(atom1,d);
                x2[d] = x(atom2,d);
                x3[d] = x(atom3,d);
                x4[d] = x(atom4,d);

                double dx;
                dx = x2[d] - x1[d];
                dx -= round(dx * inverse_L[d]) * L[d];
                r12[d] = dx;

                dx = x3[d] - x2[d];
                dx -= round(dx * inverse_L[d]) * L[d];
                r23[d] = dx;

                dx = x4[d] - x3[d];
                dx -= round(dx * inverse_L[d]) * L[d];
                r34[d] = dx;
            }

            // Compute cross products
            double n1[3], n2[3];
            n1[0] = r12[1]*r23[2] - r12[2]*r23[1];
            n1[1] = r12[2]*r23[0] - r12[0]*r23[2];
            n1[2] = r12[0]*r23[1] - r12[1]*r23[0];

            n2[0] = r23[1]*r34[2] - r23[2]*r34[1];
            n2[1] = r23[2]*r34[0] - r23[0]*r34[2];
            n2[2] = r23[0]*r34[1] - r23[1]*r34[0];

            // Compute magnitudes
            double n1_mag = sqrt(n1[0]*n1[0] + n1[1]*n1[1] + n1[2]*n1[2]);
            double n2_mag = sqrt(n2[0]*n2[0] + n2[1]*n2[1] + n2[2]*n2[2]);

            // Compute the dihedral angle phi
            double cos_phi = (n1[0]*n2[0] + n1[1]*n2[1] + n1[2]*n2[2]) / (n1_mag * n2_mag + 1e-8);
            cos_phi = fmin(fmax(cos_phi, -1.0), 1.0);  // Clamp to [-1,1]
            double sin_phi_sign = ((r12[0]*n2[0] + r12[1]*n2[1] + r12[2]*n2[2]) > 0) ? 1.0 : -1.0;
            double sin_phi = sin_phi_sign * sqrt(1.0 - cos_phi * cos_phi);
            double phi = atan2(sin_phi, cos_phi);

            // Compute the potential energy
            double potential = k1*(1 + cos(phi)) + k2*(1 - cos(2*phi)) +
                            k3*(1 + cos(3*phi)) + k4*(1 - cos(4*phi)); // The usual factor of 0.5 is already part of k

            V += potential;
        },
    result);
    return result;
}

void Bonds::force_bonds(const particles_instance& particles, type_f& f) {
    // Capture all needed members of "particles" here. We do not want to reference 
    // any members of "particle" directly inside of the kernel, as the class contains
    // functions that are not device safe. This would trigger a lot of compiler warnings.
    auto& x = particles.x;
    auto& L = particles.L;
    auto& inverse_halved_L = particles.inverse_halved_L;
    auto& bonds = this->unconstrained_bonds;
    auto& bondTypes = this->bondTypes;

    Kokkos::parallel_for(
        "compute_force_bonds",
        Kokkos::RangePolicy<>(0, bonds.extent(0)),
        KOKKOS_LAMBDA(int i) {

            int atom1 = bonds(i).atom1;
            int atom2 = bonds(i).atom2;
            int type = bonds(i).type;
            double k = bondTypes(type).k;
            double r0 = bondTypes(type).r0;

            // Compute the displacement vector
            double dx = x(atom1,0) - x(atom2,0);
            dx -= int(dx * inverse_halved_L[0]) * L[0];
            double dy = x(atom1,1) - x(atom2,1);
            dy -= int(dy * inverse_halved_L[1]) * L[1];
            double dz = x(atom1,2) - x(atom2,2);
            dz -= int(dz * inverse_halved_L[2]) * L[2];

            // Compute the squared distance
            double r2 = dx*dx + dy*dy + dz*dz;
            double r = sqrt(r2);

            // Compute the force magnitude
            double dr = r - r0;
            double force_mag = -2.0 * k * dr / r;
            
            // Apply the forces to the atoms
            for (int d = 0; d < 3; ++d) {
                double f_component =
                  force_mag * (d == 0 ? dx :
                               d == 1 ? dy : dz);
          
                Kokkos::atomic_add(&f(atom1, d), -f_component);
                Kokkos::atomic_add(&f(atom2, d),  f_component);
            }
        });
    Kokkos::fence();
}

void Bonds::force_angles(const particles_instance& particles, type_f& f) {
    // Capture all needed members of "particles" here. We do not want to reference 
    // any members of "particle" directly inside of the kernel, as the class contains
    // functions that are not device safe. This would trigger a lot of compiler warnings.
    auto& x = particles.x;
    auto& L = particles.L;
    auto& inverse_halved_L = particles.inverse_halved_L;
    auto& angles = this->angles;
    auto& angleTypes = this->angleTypes;

    Kokkos::parallel_for(
        "compute_force_angles",
        Kokkos::RangePolicy<>(0, angles.extent(0)),
        KOKKOS_LAMBDA(int i) {
                int atom1 = angles(i).atom1;
                int atom2 = angles(i).atom2;
                int atom3 = angles(i).atom3;
                int type = angles(i).type;
                double k = angleTypes(type).k;
                double theta0 = angleTypes(type).theta0;

                // Vector from atom2 to atom1
                double delx1 = x(atom1, 0) - x(atom2, 0);
                double dely1 = x(atom1, 1) - x(atom2, 1);
                double delz1 = x(atom1, 2) - x(atom2, 2);

                // Apply periodic boundary conditions
                delx1 -= int(delx1 * inverse_halved_L[0]) * L[0];
                dely1 -= int(dely1 * inverse_halved_L[1]) * L[1];
                delz1 -= int(delz1 * inverse_halved_L[2]) * L[2];

                double rsq1 = delx1 * delx1 + dely1 * dely1 + delz1 * delz1;
                double r1 = sqrt(rsq1);

                // Vector from atom2 to atom3
                double delx2 = x(atom3, 0) - x(atom2, 0);
                double dely2 = x(atom3, 1) - x(atom2, 1);
                double delz2 = x(atom3, 2) - x(atom2, 2);

                // Apply periodic boundary conditions
                delx2 -= int(delx2 * inverse_halved_L[0]) * L[0];
                dely2 -= int(dely2 * inverse_halved_L[1]) * L[1];
                delz2 -= int(delz2 * inverse_halved_L[2]) * L[2];

                double rsq2 = delx2 * delx2 + dely2 * dely2 + delz2 * delz2;
                double r2 = sqrt(rsq2);

                double c = (delx1 * delx2 + dely1 * dely2 + delz1 * delz2) / (r1 * r2);
                c = fmin(fmax(c, -1.0), 1.0);

                double dtheta = acos(c) - theta0;

                double s = sqrt(1.0 - c * c);
                double a = -2.0 * k * dtheta / (s + 1e-8);

                double f1x = a * (delx2 / r2 - delx1 * c / (r1 * r1));
                double f1y = a * (dely2 / r2 - dely1 * c / (r1 * r1));
                double f1z = a * (delz2 / r2 - delz1 * c / (r1 * r1));

                double f3x = a * (delx1 / r1 - delx2 * c / (r2 * r2));
                double f3y = a * (dely1 / r1 - dely2 * c / (r2 * r2));
                double f3z = a * (delz1 / r1 - delz2 * c / (r2 * r2));

                double f2x = -(f1x + f3x);
                double f2y = -(f1y + f3y);
                double f2z = -(f1z + f3z);

                // Accumulate forces using atomic operations to ensure thread safety
                Kokkos::atomic_add(&f(atom1, 0), f1x);
                Kokkos::atomic_add(&f(atom1, 1), f1y);
                Kokkos::atomic_add(&f(atom1, 2), f1z);

                Kokkos::atomic_add(&f(atom2, 0), f2x);
                Kokkos::atomic_add(&f(atom2, 1), f2y);
                Kokkos::atomic_add(&f(atom2, 2), f2z);

                Kokkos::atomic_add(&f(atom3, 0), f3x);
                Kokkos::atomic_add(&f(atom3, 1), f3y);
                Kokkos::atomic_add(&f(atom3, 2), f3z);
        });
    Kokkos::fence();
}

void Bonds::force_dihedrals(const particles_instance& particles, type_f& f) {
    // Capture all needed members of "particles" here. We do not want to reference 
    // any members of "particle" directly inside of the kernel, as the class contains
    // functions that are not device safe. This would trigger a lot of compiler warnings.
    auto& x = particles.x;
    auto& L = particles.L;
    auto& inverse_L = particles.inverse_L;
    auto& dihedrals = this->dihedrals;
    auto& dihedralTypes = this->dihedralTypes;

    typedef Kokkos::TeamPolicy<Tag_force_dihedrals> team_policy;
    Kokkos::parallel_for(
        "compute_force_dihedrals",
        Kokkos::RangePolicy<>(0, dihedrals.extent(0)),
        KOKKOS_LAMBDA(int i) {

                int atom1 = dihedrals(i).atom1;
                int atom2 = dihedrals(i).atom2;
                int atom3 = dihedrals(i).atom3;
                int atom4 = dihedrals(i).atom4;
                int type = dihedrals(i).type;

                double k1 = dihedralTypes(type).k1;
                double k2 = dihedralTypes(type).k2;
                double k3 = dihedralTypes(type).k3;
                double k4 = dihedralTypes(type).k4;

                // Positions of the atoms
                double x1[3], x2[3], x3[3], x4[3];

                for (int d = 0; d < 3; ++d) {
                    x1[d] = x(atom1,d);
                    x2[d] = x(atom2,d);
                    x3[d] = x(atom3,d);
                    x4[d] = x(atom4,d);
                }

                // Compute displacement vectors with periodic boundary conditions
                double r12[3], r23[3], r34[3];

                for (int d = 0; d < 3; ++d) {
                    double dx;
                    dx = x2[d] - x1[d];
                    dx -= round(dx * inverse_L[d]) * L[d];
                    r12[d] = dx;

                    dx = x3[d] - x2[d];
                    dx -= round(dx * inverse_L[d]) * L[d];
                    r23[d] = dx;

                    dx = x4[d] - x3[d];
                    dx -= round(dx * inverse_L[d]) * L[d];
                    r34[d] = dx;
                }

                // Compute bond vectors
                double b1[3], b2[3], b3[3];
                for (int d = 0; d < 3; ++d) {
                    b1[d] = r12[d];
                    b2[d] = r23[d];
                    b3[d] = r34[d];
                }

                // Compute cross products
                double c1[3], c2[3], c1_mag2, c2_mag2, b2_mag2;
                // c1 = b1 x b2
                c1[0] = b1[1]*b2[2] - b1[2]*b2[1];
                c1[1] = b1[2]*b2[0] - b1[0]*b2[2];
                c1[2] = b1[0]*b2[1] - b1[1]*b2[0];

                // c2 = b2 x b3
                c2[0] = b2[1]*b3[2] - b2[2]*b3[1];
                c2[1] = b2[2]*b3[0] - b2[0]*b3[2];
                c2[2] = b2[0]*b3[1] - b2[1]*b3[0];

                // Compute magnitudes squared
                c1_mag2 = c1[0]*c1[0] + c1[1]*c1[1] + c1[2]*c1[2];
                c2_mag2 = c2[0]*c2[0] + c2[1]*c2[1] + c2[2]*c2[2];
                b2_mag2 = b2[0]*b2[0] + b2[1]*b2[1] + b2[2]*b2[2];

                double c1_mag = sqrt(c1_mag2);
                double c2_mag = sqrt(c2_mag2);

                // Compute the dihedral angle phi
                double cos_phi = (c1[0]*c2[0] + c1[1]*c2[1] + c1[2]*c2[2]) / (c1_mag * c2_mag + 1e-8);
                cos_phi = fmin(fmax(cos_phi, -1.0), 1.0);  // Clamp to [-1,1]
                double sin_phi = b2[0]*(c1[1]*c2[2] - c1[2]*c2[1]) + b2[1]*(c1[2]*c2[0] - c1[0]*c2[2]) + b2[2]*(c1[0]*c2[1] - c1[1]*c2[0]);
                sin_phi /= (b2_mag2 * c1_mag * c2_mag + 1e-8);

                double phi = atan2(sin_phi, cos_phi);

                // Compute the derivative of the potential with respect to phi
                double dV_dphi = k1 * sin(phi) - 2.0 * k2 * sin(2.0 * phi) + 3.0 * k3 * sin(3.0 * phi) - 4.0 * k4 * sin(4.0 * phi);

                // Compute the forces on each atom
                double df[4][3];  // Forces on atoms 1 to 4
                double denom = b2_mag2 * c1_mag * c2_mag + 1e-8;

                // Auxiliary terms for force calculations
                double s1 = 1.0 / (c1_mag2 + 1e-8);
                double s2 = 1.0 / (c2_mag2 + 1e-8);

                // Calculate the gradients
                for (int d = 0; d < 3; ++d) {
                    // Terms for atoms 1 and 4
                    double term1 = c1[d] * s1 * b2_mag2;
                    double term2 = c2[d] * s2 * b2_mag2;

                    // Atom 1
                    df[0][d] = -dV_dphi * b2_mag2 * term1 / denom;

                    // Atom 4
                    df[3][d] = dV_dphi * b2_mag2 * term2 / denom;

                    // Terms for atom 2
                    double b1_dot_b2 = b1[0]*b2[0] + b1[1]*b2[1] + b1[2]*b2[2];
                    double b3_dot_b2 = b3[0]*b2[0] + b3[1]*b2[1] + b3[2]*b2[2];

                    double term3 = (b1[d] - b2[d] * b1_dot_b2 / b2_mag2) * s1;
                    double term4 = (b3[d] - b2[d] * b3_dot_b2 / b2_mag2) * s2;

                    df[1][d] = -dV_dphi * (term3 * b2_mag2 - term1 * (b1_dot_b2 + b2_mag2)) / denom;
                    df[1][d] += dV_dphi * (term4 * b2_mag2 - term2 * b3_dot_b2) / denom;

                    // Atom 3
                    df[2][d] = -df[0][d] - df[1][d] - df[3][d];
                }

                // Update forces using atomic operations
                for (int d = 0; d < 3; ++d) {
                    Kokkos::atomic_add(&f(atom1,d), df[0][d]);
                    Kokkos::atomic_add(&f(atom2,d), df[1][d]);
                    Kokkos::atomic_add(&f(atom3,d), df[2][d]);
                    Kokkos::atomic_add(&f(atom4,d), df[3][d]);
                }
        });
    Kokkos::fence();
}