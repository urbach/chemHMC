#include "SHAKE.hpp"

#include "Input_reader.hpp"
#include "bonds.hpp"


VELOCITY_VERLET_SHAKE::VELOCITY_VERLET_SHAKE(YAML::Node doc, params_class params): integrator_type(doc, params) {
    max_iter = check_and_assign_value<int>(doc["integrator"], "shake_max_iter");
    tolerance = check_and_assign_value<double>(doc["integrator"], "shake_tolerance");
}

void VELOCITY_VERLET_SHAKE::integrate() {
    find_shake_clusters();
    calc_manager->compute_force();
    Kokkos::fence();
    apply_SHAKE();
    for (size_t i = 0; i < steps; i++) {
        particles->update_momenta(dt / 2.);
        particles->update_positions(dt);
        
        calc_manager->compute_force();
        Kokkos::fence();
        apply_SHAKE();
        particles->update_momenta(dt / 2.);

        // Apply RATTLE to correct velocities
        //apply_RATTLE();
    }
}

void VELOCITY_VERLET_SHAKE::find_shake_clusters() {
    // Determine the number of constrained bonds and atoms.
    int nbonds = particles->bonds_ptr->h_constrained_bonds.extent(0);
    int N = particles->h_x.extent(0);
    
    // Setup union–find: initially each atom is its own parent.
    std::vector<int> parent(N);
    for (int i = 0; i < N; i++) {
      parent[i] = i;
    }
    
    // Helper lambda to find the root of a component.
    auto find = [&](int i, const auto &parent_ref) -> int {
      while (parent_ref[i] != i) {
        i = parent_ref[i];
      }
      return i;
    };
    
    // Helper lambda to join two sets.
    auto join = [&](int i, int j, std::vector<int> &parent_ref) {
      int ri = find(i, parent_ref);
      int rj = find(j, parent_ref);
      if (ri != rj) {
        parent_ref[ri] = rj;
      }
    };
    
    // Union all atoms connected by constrained bonds.
    for (int i = 0; i < nbonds; i++) {
      int a = particles->bonds_ptr->h_constrained_bonds(i).atom1;
      int b = particles->bonds_ptr->h_constrained_bonds(i).atom2;
      join(a, b, parent);
    }
    
    // Build clusters: map from component root to a vector containing the indices
    // of the bonds that belong to that connected component.
    std::map<int, std::vector<int>> clusters;
    for (int i = 0; i < nbonds; i++) {
      int a = particles->bonds_ptr->h_constrained_bonds(i).atom1;
      int root = find(a, parent);  // Either atom gives the same root.
      clusters[root].push_back(i);
    }
    
    // Temporary host-side containers for clusters sorted by number of bonds.
    std::vector<int> tmp_size1;                // clusters with one bond
    std::vector<std::array<int, 2>> tmp_size2;   // clusters with two bonds
    std::vector<std::array<int, 3>> tmp_size3;   // clusters with three bonds
    
    // Categorize each cluster by its number of bonds.
    for (auto &entry : clusters) {
      const std::vector<int>& bondIndices = entry.second;
      size_t nBonds = bondIndices.size();
      
      if (nBonds == 1) {
        tmp_size1.push_back(bondIndices[0]);
      } else if (nBonds == 2) {
        tmp_size2.push_back({ bondIndices[0], bondIndices[1] });
      } else if (nBonds == 3) {
        tmp_size3.push_back({ bondIndices[0], bondIndices[1], bondIndices[2] });
      } else {
        // For clusters of unexpected sizes, you can either extend the logic,
        // ignore them, or print a warning.
        std::cout << "Warning: found cluster with " << nBonds
                  << " bonds, which is not supported." << std::endl;
      }
    }
    
    // Now allocate and fill the Kokkos views.
    // Note: It is assumed that size_1_clusters, size_2_clusters, and size_3_clusters
    // are member variables of the class with the types shown below.
    
    // Allocate and copy for 1-bond clusters.
    size_1_clusters = Kokkos::View<int*>("size_1_clusters", tmp_size1.size());
    {
      auto hostView = Kokkos::create_mirror_view(Kokkos::HostSpace(), size_1_clusters);
      for (size_t i = 0; i < tmp_size1.size(); i++) {
        hostView(i) = tmp_size1[i];
      }
      Kokkos::deep_copy(size_1_clusters, hostView);
    }

    // Allocate and copy for 2-bond clusters.
    size_2_clusters = Kokkos::View<int* [2]>("size_2_clusters", tmp_size2.size());
    {
      auto hostView = Kokkos::create_mirror_view(Kokkos::HostSpace(), size_2_clusters);
      for (size_t i = 0; i < tmp_size2.size(); i++) {
        hostView(i, 0) = tmp_size2[i][0];
        hostView(i, 1) = tmp_size2[i][1];
      }
      Kokkos::deep_copy(size_2_clusters, hostView);
    }

    // Allocate and copy for 3-bond clusters.
    size_3_clusters = Kokkos::View<int* [3]>("size_3_clusters", tmp_size3.size());
    {
      auto hostView = Kokkos::create_mirror_view(Kokkos::HostSpace(), size_3_clusters);
      for (size_t i = 0; i < tmp_size3.size(); i++) {
        hostView(i, 0) = tmp_size3[i][0];
        hostView(i, 1) = tmp_size3[i][1];
        hostView(i, 2) = tmp_size3[i][2];
      }
      Kokkos::deep_copy(size_3_clusters, hostView);
    }
}

void VELOCITY_VERLET_SHAKE::generate_trial_positions() {
    auto& p = particles->p;
    auto& L = particles->L;
    auto& id = particles->id;
    auto& x = particles->x;
    auto& inverse_halved_L = particles->inverse_halved_L;
    auto& bonds = particles->bonds_ptr->constrained_bonds;
    auto& c = particles->coeff_x;    // Inverse mass lookup table
    trial_positions = Kokkos::create_mirror(x);
    Kokkos::deep_copy(trial_positions,x);

    // do an unconstrained update on all positions
    Kokkos::parallel_for(
        "SHAKE_unconstrained_update",
        Kokkos::RangePolicy<>(0, trial_positions.extent(0)),
        KOKKOS_LAMBDA(const int i) {
            for (int dir = 0; dir < 3; dir++) {
                
                trial_positions(i, dir) += this->dt * c[id[i]] * p(i, dir);
                // apply  periodic boundary condition
                trial_positions(i, dir) -= L[dir] * floor(trial_positions(i, dir) / L[dir]);
            }
        }
    );
}

void VELOCITY_VERLET_SHAKE::apply_SHAKE() {
    generate_trial_positions();
    SHAKE_size_1_cluster();
    SHAKE_size_2_cluster();
}

void VELOCITY_VERLET_SHAKE::SHAKE_size_1_cluster() {
    auto& tolerance = this->tolerance;
    auto& size_1_clusters = this->size_1_clusters;
    auto& f = particles->f;
    auto& x = particles->x;
    auto& trial_x = this->trial_positions;
    auto& L = particles->L;
    auto& id = particles->id;
    auto& inverse_halved_L = particles->inverse_halved_L;
    auto& bonds = particles->bonds_ptr->constrained_bonds;
    auto& bondTypes = particles->bonds_ptr->bondTypes;
    auto& coeff_x = particles->coeff_x;    // Inverse mass lookup table
    double dt = this->dt;

    Kokkos::parallel_for(
    "SHAKE_force_update",
    Kokkos::RangePolicy<>(0, size_1_clusters.extent(0)),
    KOKKOS_LAMBDA(const int i) {
        const int bond_idx = size_1_clusters(i);
        // Get atoms and bond type for bond i.
        int atom1 = bonds(bond_idx).atom1;
        int atom2 = bonds(bond_idx).atom2;
        int type1 = id(atom1);
        int type2 = id(atom2);
        int bondtype = bonds(bond_idx).type;
        double r0 = bondTypes(bondtype).r0;

        double m1_inv = coeff_x[type1];
        double m2_inv = coeff_x[type2];

        // distance
        double rvec[3];
        rvec[0] = x(atom1, 0) - x(atom2, 0);
        rvec[0] -= int(rvec[0] * inverse_halved_L[0]) * L[0];
        rvec[1] = x(atom1, 1) - x(atom2, 1);
        rvec[1] -= int(rvec[1] * inverse_halved_L[1]) * L[1];
        rvec[2] = x(atom1, 2) - x(atom2, 2);
        rvec[2] -= int(rvec[2] * inverse_halved_L[2]) * L[2];
        double r2 = rvec[0]*rvec[0]+rvec[1]*rvec[1]+rvec[2]*rvec[2];

        // unconstrained update distance
        double svec[3];
        svec[0] = trial_x(atom1, 0) - trial_x(atom2, 0);
        svec[0] -= int(svec[0] * inverse_halved_L[0]) * L[0];
        svec[1] = trial_x(atom1, 1) - trial_x(atom2, 1);
        svec[1] -= int(svec[1] * inverse_halved_L[1]) * L[1];
        svec[2] = trial_x(atom1, 2) - trial_x(atom2, 2);
        svec[2] -= int(svec[2] * inverse_halved_L[2]) * L[2];
        double s2 = svec[0]*svec[0]+svec[1]*svec[1]+svec[2]*svec[2];
        //Kokkos::printf("r2: %f \t s2: %f\n",r2,s2);
        // compute factors from quadratic equation
        double A = (m1_inv+m2_inv)*(m1_inv+m2_inv)*r2;
        double B = 2.0 * (m1_inv+m2_inv) * (rvec[0]*svec[0]+rvec[1]*svec[1]+rvec[2]*svec[2]);
        double C = s2 - r0*r0;

        // sanity check
        double D = B*B-4.0*A*C;
        if (D < 0.0) {
            Kokkos::printf("WARNING: Constraint determinant < 0.0!\n");
            D = 0.0;
        }

        // Compute roots of quadratic equation
        double lambda1 = (-B+sqrt(D)) / (2.0*A);
        double lambda2 = (-B-sqrt(D)) / (2.0*A);

        // Take the smaller root
        double lambda;
        if (fabs(lambda1) <= fabs(lambda2)) lambda = lambda1;
        else lambda = lambda2;

        // scale lambda so the forces have the proper magnitude when applied
        lambda /= dt*dt;

        // apply update to forces
        f(atom1,0) -= lambda*rvec[0];
        f(atom1,1) -= lambda*rvec[1];
        f(atom1,2) -= lambda*rvec[2];

        f(atom2,0) += lambda*rvec[0];
        f(atom2,1) += lambda*rvec[1];
        f(atom2,2) += lambda*rvec[2];
        }
    );
}

void VELOCITY_VERLET_SHAKE::SHAKE_size_2_cluster() {
    auto& max_iter              = this->max_iter;
    auto& tolerance             = this->tolerance;
    auto& size_2_clusters       = this->size_2_clusters;
    auto& f                     = particles->f;
    auto& x                     = particles->x;
    auto& trial_x               = this->trial_positions;
    auto& L                     = particles->L;
    auto& id                    = particles->id;
    auto& inverse_halved_L      = particles->inverse_halved_L;
    auto& bonds                 = particles->bonds_ptr->constrained_bonds;
    auto& bondTypes             = particles->bonds_ptr->bondTypes;
    auto& coeff_x               = particles->coeff_x;  // Inverse mass lookup table
    double dt_2 = this->dt * this->dt;
  
    Kokkos::parallel_for(
    "SHAKE_size_2_cluster_update",
    Kokkos::RangePolicy<>(0, size_2_clusters.extent(0)),
    KOKKOS_LAMBDA(const int i) {
        // Get the two bond indices for this cluster.
        const int bond0_idx = size_2_clusters(i, 0);
        const int bond1_idx = size_2_clusters(i, 1);

        // Retrieve atoms for each bond.
        int a = bonds(bond0_idx).atom1;
        int b = bonds(bond0_idx).atom2;
        int c = bonds(bond1_idx).atom1;
        int d = bonds(bond1_idx).atom2;

        int bondtype0 = bonds(bond0_idx).type;
        int bondtype1 = bonds(bond1_idx).type;

        int bond0_r0 = bondTypes(bondtype0).r0;
        int bond1_r0 = bondTypes(bondtype1).r0;

        // Identify the common atom. We check among the four atoms.
        int atom0 = -1, atom1 = -1, atom2 = -1;
        if (a == c || a == d) {
            atom0 = a;
            atom1 = b;
            atom2 = (a == c) ? d : c;
        } else if (b == c || b == d) {
            atom0 = b;
            atom1 = a;
            atom2 = (b == c) ? d : c;
        }
        // If for some reason no common atom is found, skip this cluster.
        if (atom0 < 0) {
            Kokkos::printf("WARNING: size 2 shake cluster without common atom loaded!");
            return;
        }

        int type0 = id(atom0);
        int type1 = id(atom1);
        int type2 = id(atom2);

        double m0_inv = coeff_x[type0];
        double m1_inv = coeff_x[type1];
        double m2_inv = coeff_x[type2];

        // atom distances
        double r01[3];
        r01[0] = x(atom0, 0) - x(atom1, 0);
        r01[0] -= int(r01[0] * inverse_halved_L[0]) * L[0];
        r01[1] = x(atom0, 1) - x(atom1, 1);
        r01[1] -= int(r01[1] * inverse_halved_L[1]) * L[1];
        r01[2] = x(atom0, 2) - x(atom1, 2);
        r01[2] -= int(r01[2] * inverse_halved_L[2]) * L[2];
        double r01_sq = r01[0]*r01[0]+r01[1]*r01[1]+r01[2]*r01[2];
        double r02[3];
        r02[0] = x(atom0, 0) - x(atom2, 0);
        r02[0] -= int(r02[0] * inverse_halved_L[0]) * L[0];
        r02[1] = x(atom0, 1) - x(atom2, 1);
        r02[1] -= int(r02[1] * inverse_halved_L[1]) * L[1];
        r02[2] = x(atom0, 2) - x(atom2, 2);
        r02[2] -= int(r02[2] * inverse_halved_L[2]) * L[2];
        double r02_sq = r02[0]*r02[0]+r02[1]*r02[1]+r02[2]*r02[2];

        // distances after unconstrained update
        double s01[3];
        s01[0] = trial_x(atom0, 0) - trial_x(atom1, 0);
        s01[0] -= int(s01[0] * inverse_halved_L[0]) * L[0];
        s01[1] = trial_x(atom0, 1) - trial_x(atom1, 1);
        s01[1] -= int(s01[1] * inverse_halved_L[1]) * L[1];
        s01[2] = trial_x(atom0, 2) - trial_x(atom1, 2);
        s01[2] -= int(s01[2] * inverse_halved_L[2]) * L[2];
        double s01_sq = s01[0]*s01[0]+s01[1]*s01[1]+s01[2]*s01[2];
        double s02[3];
        s02[0] = trial_x(atom0, 0) - trial_x(atom2, 0);
        s02[0] -= int(s02[0] * inverse_halved_L[0]) * L[0];
        s02[1] = trial_x(atom0, 1) - trial_x(atom2, 1);
        s02[1] -= int(s02[1] * inverse_halved_L[1]) * L[1];
        s02[2] = trial_x(atom0, 2) - trial_x(atom2, 2);
        s02[2] -= int(s02[2] * inverse_halved_L[2]) * L[2];
        double s02_sq = s02[0]*s02[0]+s02[1]*s02[1]+s02[2]*s02[2];
    
        double a11 = 2.0*(m0_inv+m1_inv)*(s01[0]*r01[0]+s01[1]*r01[1]+s01[2]*r01[2]);
        double a12 = 2.0*m0_inv*(s01[0]*r02[0] + s01[1]*r02[1] + s01[2]*r02[2]);
        double a21 = 2.0*m0_inv*(s02[0]*r01[0] + s02[1]*r01[1] + s02[2]*r01[2]);
        double a22 = 2.0*(m0_inv+m2_inv)*(s02[0]*r02[0] + s02[1]*r02[1] + s02[2]*r02[2]);

        double D = a11*a22 - a12*a21;
        if (D == 0.0) Kokkos::abort("Warning:Constraint determinant = 0.0\n");
        double D_inv = 1.0/D;

        double a11inv = a22*D_inv;
        double a12inv = -a12*D_inv;
        double a21inv = -a21*D_inv;
        double a22inv = a11*D_inv;

        double r0102 = (r01[0]*r02[0] + r01[1]*r02[1] + r01[2]*r02[2]);

        double quad1_0101 = (m0_inv+m1_inv)*(m0_inv+m1_inv) * r01_sq;
        double quad1_0202 = m0_inv*m0_inv * r02_sq;
        double quad1_0102 = 2.0 * (m0_inv+m1_inv)*m0_inv * r0102;

        double quad2_0202 = (m0_inv+m2_inv)*(m0_inv+m2_inv) * r02_sq;
        double quad2_0101 = m0_inv*m0_inv * r01_sq;
        double quad2_0102 = 2.0 * (m0_inv+m2_inv)*m0_inv * r0102;

        double lambda01 = 0.0;
        double lambda02 = 0.0;
        int niter = 0;
        int done = 0;

        double quad1,quad2,b1,b2,lambda01_new,lambda02_new;

        while (!done && niter < max_iter) {
            quad1 = quad1_0101 * lambda01*lambda01 + quad1_0202 * lambda02*lambda02 +
                quad1_0102 * lambda01*lambda02;
            quad2 = quad2_0101 * lambda01*lambda01 + quad2_0202 * lambda02*lambda02 +
                quad2_0102 * lambda01*lambda02;
        
            b1 = bond0_r0*bond0_r0 - s01_sq - quad1;
            b2 = bond1_r0*bond1_r0 - s02_sq - quad2;
        
            lambda01_new = a11inv*b1 + a12inv*b2;
            lambda02_new = a21inv*b1 + a22inv*b2;
        
            done = 1;
            if (fabs(lambda01_new-lambda01) > tolerance) done = 0;
            if (fabs(lambda02_new-lambda02) > tolerance) done = 0;
        
            lambda01 = lambda01_new;
            lambda02 = lambda02_new;
        
            niter++;
        }

        lambda01 /= dt_2;
        lambda02 /= dt_2;

        f(atom0,0) -= lambda01*r01[0] + lambda02*r02[0];
        f(atom0,1) -= lambda01*r01[1] + lambda02*r02[1];
        f(atom0,2) -= lambda01*r01[2] + lambda02*r02[2];

        f(atom1,0) += lambda01*r01[0];
        f(atom1,1) += lambda01*r01[1];
        f(atom1,2) += lambda01*r01[2];

        f(atom2,0) += lambda02*r02[0];
        f(atom2,1) += lambda02*r02[1];
        f(atom2,2) += lambda02*r02[2];
        }
    );
}

// RATTLE: Corrects velocities to satisfy constraints
void VELOCITY_VERLET_SHAKE::apply_RATTLE() {
    auto& max_iter = this->max_iter;  // Max iterations  
    auto& tolerance = this->tolerance; // Convergence criterion

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