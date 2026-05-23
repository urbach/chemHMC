#include "SHAKE.hpp"

#include "Input_reader.hpp"
#include "bonds.hpp"


VELOCITY_VERLET_SHAKE::VELOCITY_VERLET_SHAKE(YAML::Node doc, params_class params): integrator_type(doc, params) {
    max_iter = check_and_assign_value<int>(doc["integrator"], "shake_max_iter");
    tolerance = check_and_assign_value<double>(doc["integrator"], "shake_tolerance");
}

void VELOCITY_VERLET_SHAKE::integrate() {
    find_shake_clusters();
    apply_RATTLE();
    calc_manager->compute_force();
    Kokkos::fence();
    for (size_t i = 0; i < steps; i++) {
        particles->update_momenta(dt / 2.);
        apply_SHAKE();
        particles->neighbor_list->build_verlet_list(*particles);
        
        calc_manager->compute_force();
        Kokkos::fence();
        particles->update_momenta(dt / 2.);
        apply_RATTLE();
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
    auto& c = particles->coeff_x;    // Inverse mass lookup table
    auto& dt = this->dt;
    this->old_positions = Kokkos::View<double*[3]>("old_positions", particles->N);
    this->trial_positions = Kokkos::View<double*[3]>("trial_positions",particles->N);
    Kokkos::deep_copy(old_positions, x);
    Kokkos::deep_copy(trial_positions,x);
    auto& trial_positions = this->trial_positions;
    // do an unconstrained update on all positions
    Kokkos::parallel_for(
        "SHAKE_unconstrained_update",
        Kokkos::RangePolicy<>(0, trial_positions.extent(0)),
        KOKKOS_LAMBDA(const int i) {
            for (int dir = 0; dir < 3; dir++) {
                
                trial_positions(i, dir) += dt * c[id[i]] * p(i, dir);
                // apply  periodic boundary condition
                trial_positions(i, dir) -= L[dir] * floor(trial_positions(i, dir) / L[dir]);
            }
        }
    );
}

void VELOCITY_VERLET_SHAKE::apply_SHAKE() {
    generate_trial_positions();
    Kokkos::deep_copy(particles->x, trial_positions);
    SHAKE_size_1_cluster();
    SHAKE_size_2_cluster();
    SHAKE_size_3_cluster();
    Kokkos::fence();
}

void VELOCITY_VERLET_SHAKE::SHAKE_size_1_cluster() {
    auto& size_1_clusters = this->size_1_clusters;
    auto& x = particles->x;
    auto& old_x = this->old_positions;
    auto& p = particles->p;
    auto& trial_x = this->trial_positions;
    auto& L = particles->L;
    auto& id = particles->id;
    auto& inverse_halved_L = particles->inverse_halved_L;
    auto& bonds = particles->bonds_ptr->constrained_bonds;
    auto& bondTypes = particles->bonds_ptr->bondTypes;
    auto& coeff_x = particles->coeff_x;    // Inverse mass lookup table
    double dt = this->dt;
    Kokkos::parallel_for(
    "SHAKE_position_update",
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
        rvec[0] = old_x(atom1, 0) - old_x(atom2, 0);
        rvec[0] -= int(rvec[0] * inverse_halved_L[0]) * L[0];
        rvec[1] = old_x(atom1, 1) - old_x(atom2, 1);
        rvec[1] -= int(rvec[1] * inverse_halved_L[1]) * L[1];
        rvec[2] = old_x(atom1, 2) - old_x(atom2, 2);
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
        for (int dir = 0; dir < 3; dir++) {
            double dx1 = m1_inv * lambda * rvec[dir];
            double dx2 = -m2_inv * lambda * rvec[dir];
            x(atom1, dir) = trial_x(atom1, dir) + dx1;
            x(atom2, dir) = trial_x(atom2, dir) + dx2;
            x(atom1, dir) -= L[dir] * floor(x(atom1, dir) / L[dir]);
            x(atom2, dir) -= L[dir] * floor(x(atom2, dir) / L[dir]);
            p(atom1, dir) += lambda * rvec[dir] / dt;
            p(atom2, dir) -= lambda * rvec[dir] / dt;
        }
        }
    );
    Kokkos::fence();
}

void VELOCITY_VERLET_SHAKE::SHAKE_size_2_cluster() {
    auto& max_iter              = this->max_iter;
    auto& tolerance             = this->tolerance;
    auto& size_2_clusters       = this->size_2_clusters;
    auto& x                     = particles->x;
    auto& old_x                 = this->old_positions;
    auto& p                     = particles->p;
    auto& trial_x               = this->trial_positions;
    auto& L                     = particles->L;
    auto& id                    = particles->id;
    auto& inverse_halved_L      = particles->inverse_halved_L;
    auto& bonds                 = particles->bonds_ptr->constrained_bonds;
    auto& bondTypes             = particles->bonds_ptr->bondTypes;
    auto& coeff_x               = particles->coeff_x;  // Inverse mass lookup table
    double dt = this->dt;
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

        double bond0_r0 = bondTypes(bondtype0).r0;
        double bond1_r0 = bondTypes(bondtype1).r0;
        
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
        r01[0] = old_x(atom0, 0) - old_x(atom1, 0);
        r01[0] -= int(r01[0] * inverse_halved_L[0]) * L[0];
        r01[1] = old_x(atom0, 1) - old_x(atom1, 1);
        r01[1] -= int(r01[1] * inverse_halved_L[1]) * L[1];
        r01[2] = old_x(atom0, 2) - old_x(atom1, 2);
        r01[2] -= int(r01[2] * inverse_halved_L[2]) * L[2];
        double r01_sq = r01[0]*r01[0]+r01[1]*r01[1]+r01[2]*r01[2];
        double r02[3];
        r02[0] = old_x(atom0, 0) - old_x(atom2, 0);
        r02[0] -= int(r02[0] * inverse_halved_L[0]) * L[0];
        r02[1] = old_x(atom0, 1) - old_x(atom2, 1);
        r02[1] -= int(r02[1] * inverse_halved_L[1]) * L[1];
        r02[2] = old_x(atom0, 2) - old_x(atom2, 2);
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

        for (int dir = 0; dir < 3; dir++) {
            double impulse0 = lambda01*r01[dir] + lambda02*r02[dir];
            double impulse1 = -lambda01*r01[dir];
            double impulse2 = -lambda02*r02[dir];

            x(atom0, dir) = trial_x(atom0, dir) + m0_inv * impulse0;
            x(atom1, dir) = trial_x(atom1, dir) + m1_inv * impulse1;
            x(atom2, dir) = trial_x(atom2, dir) + m2_inv * impulse2;
            x(atom0, dir) -= L[dir] * floor(x(atom0, dir) / L[dir]);
            x(atom1, dir) -= L[dir] * floor(x(atom1, dir) / L[dir]);
            x(atom2, dir) -= L[dir] * floor(x(atom2, dir) / L[dir]);

            p(atom0, dir) += impulse0 / dt;
            p(atom1, dir) += impulse1 / dt;
            p(atom2, dir) += impulse2 / dt;
        }
        }
    );
    Kokkos::fence();
}

void VELOCITY_VERLET_SHAKE::SHAKE_size_3_cluster() {
    auto& max_iter              = this->max_iter;
    auto& tolerance             = this->tolerance;
    auto& size_3_clusters       = this->size_3_clusters;
    auto& x                     = particles->x;
    auto& old_x                 = this->old_positions;
    auto& p                     = particles->p;
    auto& trial_x               = this->trial_positions;
    auto& L                     = particles->L;
    auto& id                    = particles->id;
    auto& inverse_halved_L      = particles->inverse_halved_L;
    auto& bonds                 = particles->bonds_ptr->constrained_bonds;
    auto& bondTypes             = particles->bonds_ptr->bondTypes;
    auto& coeff_x               = particles->coeff_x;  // Inverse mass lookup table
    double dt = this->dt;
    Kokkos::parallel_for(
    "SHAKE_size_2_cluster_update",
    Kokkos::RangePolicy<>(0, size_3_clusters.extent(0)),
    KOKKOS_LAMBDA(const int i) {
        // Get the three bond indices for this size_3 cluster.
        const int bond0_idx = size_3_clusters(i, 0);
        const int bond1_idx = size_3_clusters(i, 1);
        const int bond2_idx = size_3_clusters(i, 2);

        // Retrieve atoms for each bond.
        int a = bonds(bond0_idx).atom1;
        int b = bonds(bond0_idx).atom2;
        int c = bonds(bond1_idx).atom1;
        int d = bonds(bond1_idx).atom2;
        int e = bonds(bond2_idx).atom1;
        int g = bonds(bond2_idx).atom2; // f is already the force

        // Optionally retrieve bond types if needed.
        int bondtype0 = bonds(bond0_idx).type;
        int bondtype1 = bonds(bond1_idx).type;
        int bondtype2 = bonds(bond2_idx).type;
        double bond0_r0 = bondTypes(bondtype0).r0;
        double bond1_r0 = bondTypes(bondtype1).r0;
        double bond2_r0 = bondTypes(bondtype2).r0;

        // Identify the common atom among the three bonds.
        // We check the two atoms in bond 0 to see if one of them is common in bonds 1 and 2.
        int atom0 = -1;
        int atom1 = -1, atom2 = -1, atom3 = -1;

        if ((a == c || a == d) && (a == e || a == g)) {
            atom0 = a;
            // For bond0, the peripheral atom is the one that is not the common atom.
            atom1 = b;
            // For bond1: choose the atom that is not common.
            atom2 = (c == a) ? d : c;
            // For bond2: choose the atom that is not common.
            atom3 = (e == a) ? g : e;
        } else if ((b == c || b == d) && (b == e || b == g)) {
            atom0 = b;
            atom1 = a;
            atom2 = (c == b) ? d : c;
            atom3 = (e == b) ? g : e;
        } else {
            Kokkos::printf("WARNING: size 3 shake cluster without a common atom loaded!\n");
            return;
        }

        int type0 = id(atom0);
        int type1 = id(atom1);
        int type2 = id(atom2);
        int type3 = id(atom3);

        double m0_inv = coeff_x[type0];
        double m1_inv = coeff_x[type1];
        double m2_inv = coeff_x[type2];
        double m3_inv = coeff_x[type3];

        // current distances
        double r01[3];
        r01[0] = old_x(atom0, 0) - old_x(atom1, 0);
        r01[0] -= int(r01[0] * inverse_halved_L[0]) * L[0];
        r01[1] = old_x(atom0, 1) - old_x(atom1, 1);
        r01[1] -= int(r01[1] * inverse_halved_L[1]) * L[1];
        r01[2] = old_x(atom0, 2) - old_x(atom1, 2);
        r01[2] -= int(r01[2] * inverse_halved_L[2]) * L[2];
        double r01_sq = r01[0]*r01[0]+r01[1]*r01[1]+r01[2]*r01[2];

        double r02[3];
        r02[0] = old_x(atom0, 0) - old_x(atom2, 0);
        r02[0] -= int(r02[0] * inverse_halved_L[0]) * L[0];
        r02[1] = old_x(atom0, 1) - old_x(atom2, 1);
        r02[1] -= int(r02[1] * inverse_halved_L[1]) * L[1];
        r02[2] = old_x(atom0, 2) - old_x(atom2, 2);
        r02[2] -= int(r02[2] * inverse_halved_L[2]) * L[2];
        double r02_sq = r02[0]*r02[0]+r02[1]*r02[1]+r02[2]*r02[2];

        double r03[3];
        r03[0] = old_x(atom0, 0) - old_x(atom3, 0);
        r03[0] -= int(r03[0] * inverse_halved_L[0]) * L[0];
        r03[1] = old_x(atom0, 1) - old_x(atom3, 1);
        r03[1] -= int(r03[1] * inverse_halved_L[1]) * L[1];
        r03[2] = old_x(atom0, 2) - old_x(atom3, 2);
        r03[2] -= int(r03[2] * inverse_halved_L[2]) * L[2];
        double r03_sq = r03[0]*r03[0]+r03[1]*r03[1]+r03[2]*r03[2];

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

        double s03[3];
        s03[0] = trial_x(atom0, 0) - trial_x(atom3, 0);
        s03[0] -= int(s03[0] * inverse_halved_L[0]) * L[0];
        s03[1] = trial_x(atom0, 1) - trial_x(atom3, 1);
        s03[1] -= int(s03[1] * inverse_halved_L[1]) * L[1];
        s03[2] = trial_x(atom0, 2) - trial_x(atom3, 2);
        s03[2] -= int(s03[2] * inverse_halved_L[2]) * L[2];
        double s03_sq = s03[0]*s03[0]+s03[1]*s03[1]+s03[2]*s03[2];
    
        double a11 = 2.0 * (m0_inv+m1_inv)*(s01[0]*r01[0]+s01[1]*r01[1]+s01[2]*r01[2]);
        double a12 = 2.0 * m0_inv*(s01[0]*r02[0]+s01[1]*r02[1]+s01[2]*r02[2]);
        double a13 = 2.0 * m0_inv*(s01[0]*r03[0]+s01[1]*r03[1]+s01[2]*r03[2]);
        double a21 = 2.0 * m0_inv*(s02[0]*r01[0]+s02[1]*r01[1]+s02[2]*r01[2]);
        double a22 = 2.0 * (m0_inv+m2_inv)*(s02[0]*r02[0]+s02[1]*r02[1]+s02[2]*r02[2]);
        double a23 = 2.0 * m0_inv*(s02[0]*r03[0]+s02[1]*r03[1]+s02[2]*r03[2]);
        double a31 = 2.0 * m0_inv*(s03[0]*r01[0]+s03[1]*r01[1]+s03[2]*r01[2]);
        double a32 = 2.0 * m0_inv*(s03[0]*r02[0]+s03[1]*r02[1]+s03[2]*r02[2]);
        double a33 = 2.0 * (m0_inv+m3_inv)*(s03[0]*r03[0]+s03[1]*r03[1]+s03[2]*r03[2]);

        double determ = a11*a22*a33 + a12*a23*a31 + a13*a21*a32-a11*a23*a32 - a12*a21*a33 - a13*a22*a31;
        if (determ == 0.0) Kokkos::abort("Contraint determinant = 0.0");
        double D_inv = 1.0/determ;

        double a11inv = D_inv * (a22*a33 - a23*a32);
        double a12inv = -D_inv * (a12*a33 - a13*a32);
        double a13inv = D_inv * (a12*a23 - a13*a22);
        double a21inv = -D_inv * (a21*a33 - a23*a31);
        double a22inv = D_inv * (a11*a33 - a13*a31);
        double a23inv = -D_inv * (a11*a23 - a13*a21);
        double a31inv = D_inv * (a21*a32 - a22*a31);
        double a32inv = -D_inv * (a11*a32 - a12*a31);
        double a33inv = D_inv * (a11*a22 - a12*a21);

        // compute coeffs

        double r0102 = (r01[0]*r02[0] + r01[1]*r02[1] + r01[2]*r02[2]);
        double r0103 = (r01[0]*r03[0] + r01[1]*r03[1] + r01[2]*r03[2]);
        double r0203 = (r02[0]*r03[0] + r02[1]*r03[1] + r02[2]*r03[2]);

        double quad1_0101 = (m0_inv+m1_inv)*(m0_inv+m1_inv) * r01_sq;
        double quad1_0202 = m0_inv*m0_inv * r02_sq;
        double quad1_0303 = m0_inv*m0_inv * r03_sq;
        double quad1_0102 = 2.0 * (m0_inv+m1_inv)*m0_inv * r0102;
        double quad1_0103 = 2.0 * (m0_inv+m1_inv)*m0_inv * r0103;
        double quad1_0203 = 2.0 * m0_inv*m0_inv * r0203;

        double quad2_0101 = m0_inv*m0_inv * r01_sq;
        double quad2_0202 = (m0_inv+m2_inv)*(m0_inv+m2_inv) * r02_sq;
        double quad2_0303 = m0_inv*m0_inv * r03_sq;
        double quad2_0102 = 2.0 * (m0_inv+m2_inv)*m0_inv * r0102;
        double quad2_0103 = 2.0 * m0_inv*m0_inv * r0103;
        double quad2_0203 = 2.0 * (m0_inv+m2_inv)*m0_inv * r0203;

        double quad3_0101 = m0_inv*m0_inv * r01_sq;
        double quad3_0202 = m0_inv*m0_inv * r02_sq;
        double quad3_0303 = (m0_inv+m3_inv)*(m0_inv+m3_inv) * r03_sq;
        double quad3_0102 = 2.0 * m0_inv*m0_inv * r0102;
        double quad3_0103 = 2.0 * (m0_inv+m3_inv)*m0_inv * r0103;
        double quad3_0203 = 2.0 * (m0_inv+m3_inv)*m0_inv * r0203;

        double lambda01 = 0.0;
        double lambda02 = 0.0;
        double lambda03 = 0.0;
        int niter = 0;
        int done = 0;

        double quad1,quad2,quad3,b1,b2,b3,lambda01_new,lambda02_new,lambda03_new;

        while (!done && niter < max_iter) {
            quad1 = quad1_0101 * lambda01*lambda01 +
            quad1_0202 * lambda02*lambda02 +
            quad1_0303 * lambda03*lambda03 +
            quad1_0102 * lambda01*lambda02 +
            quad1_0103 * lambda01*lambda03 +
            quad1_0203 * lambda02*lambda03;

            quad2 = quad2_0101 * lambda01*lambda01 +
            quad2_0202 * lambda02*lambda02 +
            quad2_0303 * lambda03*lambda03 +
            quad2_0102 * lambda01*lambda02 +
            quad2_0103 * lambda01*lambda03 +
            quad2_0203 * lambda02*lambda03;

            quad3 = quad3_0101 * lambda01*lambda01 +
            quad3_0202 * lambda02*lambda02 +
            quad3_0303 * lambda03*lambda03 +
            quad3_0102 * lambda01*lambda02 +
            quad3_0103 * lambda01*lambda03 +
            quad3_0203 * lambda02*lambda03;

            b1 = bond0_r0*bond0_r0 - s01_sq - quad1;
            b2 = bond1_r0*bond1_r0 - s02_sq - quad2;
            b3 = bond2_r0*bond2_r0 - s03_sq - quad3;

            lambda01_new = a11inv*b1 + a12inv*b2 + a13inv*b3;
            lambda02_new = a21inv*b1 + a22inv*b2 + a23inv*b3;
            lambda03_new = a31inv*b1 + a32inv*b2 + a33inv*b3;

            done = 1;
            if (fabs(lambda01_new-lambda01) > tolerance) done = 0;
            if (fabs(lambda02_new-lambda02) > tolerance) done = 0;
            if (fabs(lambda03_new-lambda03) > tolerance) done = 0;

            lambda01 = lambda01_new;
            lambda02 = lambda02_new;
            lambda03 = lambda03_new;

            niter++;
        }
        
        for (int dir = 0; dir < 3; dir++) {
            double impulse0 = lambda01*r01[dir] + lambda02*r02[dir] + lambda03*r03[dir];
            double impulse1 = -lambda01*r01[dir];
            double impulse2 = -lambda02*r02[dir];
            double impulse3 = -lambda03*r03[dir];

            x(atom0, dir) = trial_x(atom0, dir) + m0_inv * impulse0;
            x(atom1, dir) = trial_x(atom1, dir) + m1_inv * impulse1;
            x(atom2, dir) = trial_x(atom2, dir) + m2_inv * impulse2;
            x(atom3, dir) = trial_x(atom3, dir) + m3_inv * impulse3;
            x(atom0, dir) -= L[dir] * floor(x(atom0, dir) / L[dir]);
            x(atom1, dir) -= L[dir] * floor(x(atom1, dir) / L[dir]);
            x(atom2, dir) -= L[dir] * floor(x(atom2, dir) / L[dir]);
            x(atom3, dir) -= L[dir] * floor(x(atom3, dir) / L[dir]);

            p(atom0, dir) += impulse0 / dt;
            p(atom1, dir) += impulse1 / dt;
            p(atom2, dir) += impulse2 / dt;
            p(atom3, dir) += impulse3 / dt;
        }
        }
    );
}

// RATTLE: Corrects velocities to satisfy constraints
void VELOCITY_VERLET_SHAKE::apply_RATTLE() {
    generate_trial_momenta();
    RATTLE_size_1_cluster();
    RATTLE_size_2_cluster();
    RATTLE_size_3_cluster();
    Kokkos::fence();
}

void VELOCITY_VERLET_SHAKE::generate_trial_momenta() {
    auto& p = particles->p;
    this->trial_momenta = Kokkos::View<double*[3]>("trial_momenta",particles->N);
    Kokkos::deep_copy(trial_momenta,p);
    Kokkos::fence();
}

void VELOCITY_VERLET_SHAKE::RATTLE_size_1_cluster() {
    auto& size_1_clusters = this->size_1_clusters;
    auto& p = particles->p;
    auto& x = particles->x;
    auto& L = particles->L;
    auto& id = particles->id;
    auto& trial_p = this->trial_momenta;
    auto& inverse_halved_L = particles->inverse_halved_L;
    auto& bonds = particles->bonds_ptr->constrained_bonds;
    auto& bondTypes = particles->bonds_ptr->bondTypes;
    auto& coeff_x = particles->coeff_x;    // Inverse mass lookup table

    Kokkos::parallel_for(
    "RATTLE_force_update",
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
        double pvec[3];
        //Kokkos::printf("atom1: %f %f %f\n",trial_p(atom1, 0),trial_p(atom1, 1),trial_p(atom1, 2));
        pvec[0] = trial_p(atom1, 0)*m1_inv - trial_p(atom2, 0)*m2_inv;
        pvec[1] = trial_p(atom1, 1)*m1_inv - trial_p(atom2, 1)*m2_inv;
        pvec[2] = trial_p(atom1, 2)*m1_inv - trial_p(atom2, 2)*m2_inv;


        // compute factors for lagrange multiplier
        double A = (rvec[0]*pvec[0]+rvec[1]*pvec[1]+rvec[2]*pvec[2]);
        double B = r2 * (m1_inv+m2_inv);

        double lambda = -A/B;

        Kokkos::atomic_fetch_add(&p(atom1,0),  lambda * rvec[0]);
        Kokkos::atomic_fetch_add(&p(atom1,1),  lambda * rvec[1]);
        Kokkos::atomic_fetch_add(&p(atom1,2),  lambda * rvec[2]);

        Kokkos::atomic_fetch_add(&p(atom2,0), -lambda * rvec[0]);
        Kokkos::atomic_fetch_add(&p(atom2,1), -lambda * rvec[1]);
        Kokkos::atomic_fetch_add(&p(atom2,2), -lambda * rvec[2]);
        }
    );
}

void VELOCITY_VERLET_SHAKE::RATTLE_size_2_cluster() {
    auto& size_2_clusters       = this->size_2_clusters;
    auto& p                    = particles->p;
    auto& x                     = particles->x;
    auto& L                     = particles->L;
    auto& id                    = particles->id;
    auto& trial_p               = this->trial_momenta;
    auto& inverse_halved_L      = particles->inverse_halved_L;
    auto& bonds                 = particles->bonds_ptr->constrained_bonds;
    auto& bondTypes             = particles->bonds_ptr->bondTypes;
    auto& coeff_x               = particles->coeff_x;  // Inverse mass lookup table
    double dt_2 = this->dt * this->dt;
  
    Kokkos::parallel_for(
    "RATTLE_size_2_cluster_update",
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

        double bond0_r0 = bondTypes(bondtype0).r0;
        double bond1_r0 = bondTypes(bondtype1).r0;

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

        double p01[3];
        p01[0] = trial_p(atom0, 0)*m0_inv - trial_p(atom1, 0)*m1_inv;
        p01[1] = trial_p(atom0, 1)*m0_inv - trial_p(atom1, 1)*m1_inv;
        p01[2] = trial_p(atom0, 2)*m0_inv - trial_p(atom1, 2)*m1_inv;

        double p02[3];
        p02[0] = trial_p(atom0, 0)*m0_inv - trial_p(atom2, 0)*m2_inv;
        p02[1] = trial_p(atom0, 1)*m0_inv - trial_p(atom2, 1)*m2_inv;
        p02[2] = trial_p(atom0, 2)*m0_inv - trial_p(atom2, 2)*m2_inv;

        double A[2][2];

        A[0][0] = (m0_inv+m1_inv) * r01_sq;
        A[1][0] = (m0_inv) * (r01[0]*r02[0]+r01[1]*r02[1]+r01[2]*r02[2]);
        A[0][1] = A[1][0];
        A[1][1] = (m0_inv+m2_inv) * r02_sq;

        double cvec[2];

        cvec[0] = -(p01[0]*r01[0]+p01[1]*r01[1]+p01[2]*r01[2]);
        cvec[1] = -(p02[0]*r02[0]+p02[1]*r02[1]+p02[2]*r02[2]);

        double lambda01,lambda02,D,D_inv;

        D = A[0][0] * A[1][1] - A[0][1] * A[1][0];
        D_inv = 1.0/D;

        lambda01 = D_inv * (A[1][1] * cvec[0] - A[0][1] * cvec[1]);
        lambda02 = D_inv * (-A[1][0] * cvec[0] + A[0][0] * cvec[1]);

        // apply constraint forces
        Kokkos::atomic_fetch_add(&p(atom0,0),lambda01*r01[0] + lambda02*r02[0]);
        Kokkos::atomic_fetch_add(&p(atom0,1),lambda01*r01[1] + lambda02*r02[1]);
        Kokkos::atomic_fetch_add(&p(atom0,2),lambda01*r01[2] + lambda02*r02[2]);

        Kokkos::atomic_fetch_add(&p(atom1,0),-lambda01*r01[0]);
        Kokkos::atomic_fetch_add(&p(atom1,1),-lambda01*r01[1]);
        Kokkos::atomic_fetch_add(&p(atom1,2),-lambda01*r01[2]);

        Kokkos::atomic_fetch_add(&p(atom2,0),-lambda02*r02[0]);
        Kokkos::atomic_fetch_add(&p(atom2,1),-lambda02*r02[1]);
        Kokkos::atomic_fetch_add(&p(atom2,2),-lambda02*r02[2]);
        }
    );
}

void VELOCITY_VERLET_SHAKE::RATTLE_size_3_cluster() {
    auto& size_3_clusters       = this->size_3_clusters;
    auto& p                     = particles->p;
    auto& x                     = particles->x;
    auto& L                     = particles->L;
    auto& id                    = particles->id;
    auto& trial_p               = this->trial_momenta;
    auto& inverse_halved_L      = particles->inverse_halved_L;
    auto& bonds                 = particles->bonds_ptr->constrained_bonds;
    auto& bondTypes             = particles->bonds_ptr->bondTypes;
    auto& coeff_x               = particles->coeff_x;  // Inverse mass lookup table
  
    Kokkos::parallel_for(
    "RATTLE_size_2_cluster_update",
    Kokkos::RangePolicy<>(0, size_3_clusters.extent(0)),
    KOKKOS_LAMBDA(const int i) {
        // Get the three bond indices for this size_3 cluster.
        const int bond0_idx = size_3_clusters(i, 0);
        const int bond1_idx = size_3_clusters(i, 1);
        const int bond2_idx = size_3_clusters(i, 2);

        // Retrieve atoms for each bond.
        int a = bonds(bond0_idx).atom1;
        int b = bonds(bond0_idx).atom2;
        int c = bonds(bond1_idx).atom1;
        int d = bonds(bond1_idx).atom2;
        int e = bonds(bond2_idx).atom1;
        int g = bonds(bond2_idx).atom2; // f is already the force

        // Optionally retrieve bond types if needed.
        int bondtype0 = bonds(bond0_idx).type;
        int bondtype1 = bonds(bond1_idx).type;
        int bondtype2 = bonds(bond2_idx).type;
        double bond0_r0 = bondTypes(bondtype0).r0;
        double bond1_r0 = bondTypes(bondtype1).r0;
        double bond2_r0 = bondTypes(bondtype2).r0;

        // Identify the common atom among the three bonds.
        // We check the two atoms in bond 0 to see if one of them is common in bonds 1 and 2.
        int atom0 = -1;
        int atom1 = -1, atom2 = -1, atom3 = -1;

        if ((a == c || a == d) && (a == e || a == g)) {
            atom0 = a;
            // For bond0, the peripheral atom is the one that is not the common atom.
            atom1 = b;
            // For bond1: choose the atom that is not common.
            atom2 = (c == a) ? d : c;
            // For bond2: choose the atom that is not common.
            atom3 = (e == a) ? g : e;
        } else if ((b == c || b == d) && (b == e || b == g)) {
            atom0 = b;
            atom1 = a;
            atom2 = (c == b) ? d : c;
            atom3 = (e == b) ? g : e;
        } else {
            Kokkos::printf("WARNING: size 3 shake cluster without a common atom loaded!\n");
            return;
        }

        int type0 = id(atom0);
        int type1 = id(atom1);
        int type2 = id(atom2);
        int type3 = id(atom3);

        double m0_inv = coeff_x[type0];
        double m1_inv = coeff_x[type1];
        double m2_inv = coeff_x[type2];
        double m3_inv = coeff_x[type3];

        // current distances
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

        double r03[3];
        r03[0] = x(atom0, 0) - x(atom3, 0);
        r03[0] -= int(r03[0] * inverse_halved_L[0]) * L[0];
        r03[1] = x(atom0, 1) - x(atom3, 1);
        r03[1] -= int(r03[1] * inverse_halved_L[1]) * L[1];
        r03[2] = x(atom0, 2) - x(atom3, 2);
        r03[2] -= int(r03[2] * inverse_halved_L[2]) * L[2];
        double r03_sq = r03[0]*r03[0]+r03[1]*r03[1]+r03[2]*r03[2];

        // distances after unconstrained update
        double p01[3];
        p01[0] = trial_p(atom0, 0)*m0_inv - trial_p(atom1, 0)*m1_inv;
        p01[1] = trial_p(atom0, 1)*m0_inv - trial_p(atom1, 1)*m1_inv;
        p01[2] = trial_p(atom0, 2)*m0_inv - trial_p(atom1, 2)*m1_inv;

        double p02[3];
        p02[0] = trial_p(atom0, 0)*m0_inv - trial_p(atom2, 0)*m2_inv;
        p02[1] = trial_p(atom0, 1)*m0_inv - trial_p(atom2, 1)*m2_inv;
        p02[2] = trial_p(atom0, 2)*m0_inv - trial_p(atom2, 2)*m2_inv;

        double p03[3];
        p03[0] = trial_p(atom0, 0)*m0_inv - trial_p(atom3, 0)*m3_inv;
        p03[1] = trial_p(atom0, 1)*m0_inv - trial_p(atom3, 1)*m3_inv;
        p03[2] = trial_p(atom0, 2)*m0_inv - trial_p(atom3, 2)*m3_inv;

        double A[3][3];

        A[0][0] = (m0_inv + m1_inv) * r01_sq;
        A[0][1] = m0_inv * (r01[0]*r02[0]+r01[1]*r02[1]+r01[2]*r02[2]);
        A[1][0] = A[0][1];
        A[0][2] = m0_inv * (r01[0]*r03[0]+r01[1]*r03[1]+r01[2]*r03[2]);
        A[2][0] = A[0][2];
        A[1][1] = (m0_inv + m2_inv) * r02_sq;
        A[1][2] = m0_inv * (r02[0]*r03[0]+r02[1]*r03[1]+r02[2]*r03[2]);
        A[2][1] = A[1][2];
        A[2][2] = (m0_inv + m3_inv) * r03_sq;

        double cvec[3];

        cvec[0] = -(p01[0]*r01[0]+p01[1]*r01[1]+p01[2]*r01[2]);
        cvec[1] = -(p02[0]*r02[0]+p02[1]*r02[1]+p02[2]*r02[2]);
        cvec[2] = -(p03[0]*r03[0]+p03[1]*r03[1]+p03[2]*r03[2]);

        
        double D,D_inv;

        D = A[0][0]*A[1][1]*A[2][2] + A[0][1]*A[1][2]*A[2][0] +
            A[0][2]*A[1][0]*A[2][1] - A[0][0]*A[1][2]*A[2][1] -
            A[0][1]*A[1][0]*A[2][2] - A[0][2]*A[1][1]*A[2][0];

        if (D == 0.0) Kokkos::abort("WARNING: Rattle determinant in size 3 cluster is 0!");
        D_inv = 1.0/D;

        double A_inv[3][3];

        A_inv[0][0] =  D_inv * (A[1][1]*A[2][2] - A[1][2]*A[2][1]);
        A_inv[0][1] = -D_inv * (A[0][1]*A[2][2] - A[0][2]*A[2][1]);
        A_inv[0][2] =  D_inv * (A[0][1]*A[1][2] - A[0][2]*A[1][1]);
        A_inv[1][0] = -D_inv * (A[1][0]*A[2][2] - A[1][2]*A[2][0]);
        A_inv[1][1] =  D_inv * (A[0][0]*A[2][2] - A[0][2]*A[2][0]);
        A_inv[1][2] = -D_inv * (A[0][0]*A[1][2] - A[0][2]*A[1][0]);
        A_inv[2][0] =  D_inv * (A[1][0]*A[2][1] - A[1][1]*A[2][0]);
        A_inv[2][1] = -D_inv * (A[0][0]*A[2][1] - A[0][1]*A[2][0]);
        A_inv[2][2] =  D_inv * (A[0][0]*A[1][1] - A[0][1]*A[1][0]);

        double lambda01,lambda02,lambda03;

        lambda01 = A_inv[0][0]*cvec[0]+A_inv[0][1]*cvec[1]+A_inv[0][2]*cvec[2];
        lambda02 = A_inv[1][0]*cvec[0]+A_inv[1][1]*cvec[1]+A_inv[1][2]*cvec[2];
        lambda03 = A_inv[2][0]*cvec[0]+A_inv[2][1]*cvec[1]+A_inv[2][2]*cvec[2];

        // apply constraint forces
        Kokkos::atomic_fetch_add(&p(atom0,0), lambda01*r01[0] + lambda02*r02[0] + lambda03*r03[0]);
        Kokkos::atomic_fetch_add(&p(atom0,1), lambda01*r01[1] + lambda02*r02[1] + lambda03*r03[1]);
        Kokkos::atomic_fetch_add(&p(atom0,2), lambda01*r01[2] + lambda02*r02[2] + lambda03*r03[2]);

        Kokkos::atomic_fetch_add(&p(atom1,0), -lambda01*r01[0]);
        Kokkos::atomic_fetch_add(&p(atom1,1), -lambda01*r01[1]);
        Kokkos::atomic_fetch_add(&p(atom1,2), -lambda01*r01[2]);

        Kokkos::atomic_fetch_add(&p(atom2,0), -lambda02*r02[0]);
        Kokkos::atomic_fetch_add(&p(atom2,1), -lambda02*r02[1]);
        Kokkos::atomic_fetch_add(&p(atom2,2), -lambda02*r02[2]);

        Kokkos::atomic_fetch_add(&p(atom3,0), -lambda03*r03[0]);
        Kokkos::atomic_fetch_add(&p(atom3,1), -lambda03*r03[1]);
        Kokkos::atomic_fetch_add(&p(atom3,2), -lambda03*r03[2]);
        }
    );
}