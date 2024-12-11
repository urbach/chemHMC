#include "global.hpp"
#include "atom.hpp"
#include "particles.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

void particles_instance::read_bonds_angles(const std::string& filename) {
    std::ifstream infile(filename);
    std::string line;

    int numBonds = 0, numAngles = 0, numDihedrals = 0;
    int numBondTypes = 0, numAngleTypes = 0, numDihedralTypes;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string keyword;
        int count;
        
        // Read the entire line and check for keywords
        if (iss >> count) {
            std::getline(iss, keyword);  // Get the rest of the line as the keyword

            // Trim leading whitespace from keyword
            keyword = keyword.substr(keyword.find_first_not_of(" \t"));

            if (keyword == "bonds") {
                numBonds = count;
            } else if (keyword == "angles") {
                numAngles = count;
            } else if (keyword == "dihedrals") {
                numDihedrals = count;
            } else if (keyword == "atom types") {
                // Just skipping atom types for now
            } else if (keyword == "bond types") {
                numBondTypes = count;
            } else if (keyword == "angle types") {
                numAngleTypes = count;
            } else if (keyword == "dihedral types") {
                numDihedralTypes = count;
            }
        }
    }

    bonds = Kokkos::View<Bond*>("bonds", numBonds);
    angles = Kokkos::View<Angle*>("angles", numAngles);
    dihedrals = Kokkos::View<Dihedral*>("dihedrals", numDihedrals);
    bondTypes = Kokkos::View<BondType*>("bondTypes", numBondTypes);
    angleTypes = Kokkos::View<AngleType*>("angleTypes", numAngleTypes);
    dihedralTypes = Kokkos::View<DihedralType*>("dihedralTypes", numDihedralTypes);

    h_bonds = Kokkos::create_mirror_view(bonds);
    h_angles = Kokkos::create_mirror_view(angles);
    h_dihedrals = Kokkos::create_mirror_view(dihedrals);
    h_bondTypes = Kokkos::create_mirror_view(bondTypes);
    h_angleTypes = Kokkos::create_mirror_view(angleTypes);
    h_dihedralTypes = Kokkos::create_mirror_view(dihedralTypes);

    bool inBondSection = false, inAngleSection = false, inDihedralSection = false;
    bool inBondTypeSection = false, inAngleTypeSection = false, inDihedralTypeSection = false;

    int bondIndex = 0, angleIndex = 0, dihedralIndex = 0;
    int bondTypeIndex = 0, angleTypeIndex = 0, dihedralTypeIndex = 0;

    infile.clear();  // Reset the stream to start reading again
    infile.seekg(0); // Go back to the beginning of the file

    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);

        // Section identification
        if (line.find("Bond Coeffs") != std::string::npos) {
            inBondTypeSection = true;
            inAngleTypeSection = false;
            inDihedralTypeSection = false;
            inDihedralSection = false;
            inBondSection = false;
            inAngleSection = false;
            continue;
        }
        if (line.find("Angle Coeffs") != std::string::npos) {
            inAngleTypeSection = true;
            inBondTypeSection = false;
            inDihedralTypeSection = false;
            inDihedralSection = false;
            inBondSection = false;
            inAngleSection = false;
            continue;
        }
        if (line.find("Dihedral Coeffs") != std::string::npos) {
            inAngleTypeSection = false;
            inBondTypeSection = false;
            inDihedralTypeSection = true;
            inDihedralSection = false;
            inBondSection = false;
            inAngleSection = false;
            continue;
        }
        if (line.find("Bonds") != std::string::npos) {
            inBondSection = true;
            inAngleSection = false;
            inDihedralTypeSection = false;
            inDihedralSection = false;
            inBondTypeSection = false;
            inAngleTypeSection = false;
            continue;
        }
        if (line.find("Angles") != std::string::npos) {
            inAngleSection = true;
            inBondSection = false;
            inDihedralTypeSection = false;
            inDihedralSection = false;
            inBondTypeSection = false;
            inAngleTypeSection = false;
            continue;
        }
        if (line.find("Dihedrals") != std::string::npos) {
            inBondSection = false;
            inAngleSection = false;
            inDihedralTypeSection = false;
            inDihedralSection = true;
            inBondTypeSection = false;
            inAngleTypeSection = false;
            continue;
        }


        // Parse bond type data
        if (inBondTypeSection && bondTypeIndex < numBondTypes) {
            int type;
            double k, r0;
            if (iss >> type >> k >> r0) {
                h_bondTypes(bondTypeIndex).type = type;
                h_bondTypes(bondTypeIndex).k = k*kcaltointernal; // convert from kcal/mol to internal units
                h_bondTypes(bondTypeIndex).r0 = r0;
                bondTypeIndex++;
            }
        }

        // Parse angle type data
        if (inAngleTypeSection && angleTypeIndex < numAngleTypes) {
            int type;
            double k, theta0;
            if (iss >> type >> k >> theta0) {
                h_angleTypes(angleTypeIndex).type = type;
                h_angleTypes(angleTypeIndex).k = k*kcaltointernal;// convert from kcal/mol to internal units
                h_angleTypes(angleTypeIndex).theta0 = theta0 * M_PI/180.0;
                angleTypeIndex++;
            }
        }

        // Parse dihedral type data
        if (inDihedralTypeSection && dihedralTypeIndex < numDihedralTypes) {
            int type;
            double k1, k2, k3, k4;
            if (iss >> type >> k1 >> k2 >> k3 >> k4) {
                h_dihedralTypes(dihedralTypeIndex).type = type;
                // for some reason lammps files include the usual factor of 0.5 into all k-values
                // except for the dihedrals so we have to explicitly add it here
                h_dihedralTypes(dihedralTypeIndex).k1 = 0.5*k1*kcaltointernal;// convert from kcal/mol to internal units
                h_dihedralTypes(dihedralTypeIndex).k2 = 0.5*k2*kcaltointernal;// convert from kcal/mol to internal units
                h_dihedralTypes(dihedralTypeIndex).k3 = 0.5*k3*kcaltointernal;// convert from kcal/mol to internal units
                h_dihedralTypes(dihedralTypeIndex).k4 = 0.5*k4*kcaltointernal;// convert from kcal/mol to internal units
                dihedralTypeIndex++;
            }
        }

        // Parse bond data
        if (inBondSection && bondIndex < numBonds) {
            int id, type, atom1, atom2;
            if (iss >> id >> type >> atom1 >> atom2) {
                h_bonds(bondIndex).id = id;
                h_bonds(bondIndex).type = type;
                h_bonds(bondIndex).atom1 = atom1;
                h_bonds(bondIndex).atom2 = atom2;
                bondIndex++;
            }
        }

        // Parse angle data
        if (inAngleSection && angleIndex < numAngles) {
            int id, type, atom1, atom2, atom3;
            if (iss >> id >> type >> atom1 >> atom2 >> atom3) {
                h_angles(angleIndex).id = id;
                h_angles(angleIndex).type = type;
                h_angles(angleIndex).atom1 = atom1;
                h_angles(angleIndex).atom2 = atom2;
                h_angles(angleIndex).atom3 = atom3;
                angleIndex++;
            }
        }

        // Parse dihedral data
        if (inDihedralSection && dihedralIndex < numDihedrals) {
            int id, type, atom1, atom2, atom3, atom4;
            if (iss >> id >> type >> atom1 >> atom2 >> atom3 >> atom4) {
                h_dihedrals(dihedralIndex).id = id;
                h_dihedrals(dihedralIndex).type = type;
                h_dihedrals(dihedralIndex).atom1 = atom1;
                h_dihedrals(dihedralIndex).atom2 = atom2;
                h_dihedrals(dihedralIndex).atom3 = atom3;
                h_dihedrals(dihedralIndex).atom4 = atom4;
                dihedralIndex++;
            }
        }
    }

    // Copy data to device
    Kokkos::deep_copy(bonds, h_bonds);
    Kokkos::deep_copy(angles, h_angles);
    Kokkos::deep_copy(dihedrals, h_dihedrals);
    Kokkos::deep_copy(bondTypes, h_bondTypes);
    Kokkos::deep_copy(angleTypes, h_angleTypes);
    Kokkos::deep_copy(dihedralTypes, h_dihedralTypes);

    /*std::cout << "Number of Bonds: " << numBonds << std::endl;
    std::cout << "Number of Angles: " << numAngles << std::endl;
    std::cout << "Number of Dihedrals: " << numDihedrals << std::endl;
    std::cout << "Actual Bonds Read: " << bondIndex << std::endl;
    std::cout << "Actual Angles Read: " << angleIndex << std::endl;
    std::cout << "Actual Dihedrals Read: " << dihedralIndex << std::endl;
    std::cout << "Number of Bond Types: " << numBondTypes << std::endl;
    std::cout << "Number of Angle Types: " << numAngleTypes << std::endl;
    std::cout << "Number of Dihedral Types: " << numDihedralTypes << std::endl;

    for (int i = 0; i < bonds.extent(0); i++) {
        printf("bond number: %d \n", i);
        printf("bond type: %d \n", h_bonds(i).type);
        printf("atom1: %d atom2: %d k: %f r0: %f\n", h_bonds(i).atom1-1, h_bonds(i).atom2-1, h_bondTypes(h_bonds(i).type-1).k, h_bondTypes(h_bonds(i).type-1).r0);
    }
    
    for (int i = 0; i < bondTypes.extent(0); i++) {
        printf("bond type number: %d \n", i);
        printf("type: %d k: %f r0: %f\n", h_bondTypes(i).type, h_bondTypes(i).k, h_bondTypes(i).r0);
    }
    
    for (int i = 0; i < angles.extent(0); i++) {
        printf("angle number: %d \n", i);
        printf("atom1: %d atom2: %d atom3: %d k: %f theta0: %f\n", h_angles(i).atom1, h_angles(i).atom2, h_angles(i).atom3, h_angleTypes(h_angles(i).type-1).k, h_angleTypes(h_angles(i).type-1).theta0);
    }
    printf("EXTENT: %d", dihedrals.extent(0));
    for (int i = 0; i < dihedrals.extent(0); i++) {
        printf("dihedral number: %d \n", i);
        printf("atom1: %d atom2: %d atom3: %d atom4: %d\n", h_dihedrals(i).atom1, h_dihedrals(i).atom2, h_dihedrals(i).atom3, h_dihedrals(i).atom4);
    }

    for (int i = 0; i < dihedralTypes.extent(0); i++) {
        printf("dihedral number: %d \n", i);
        printf("atom1: %f atom2: %f atom3: %f atom4: %f\n", h_dihedralTypes(i).k1, h_dihedralTypes(i).k2, h_dihedralTypes(i).k3, h_dihedralTypes(i).k4);
    }*/

}

void particles_instance::compute_force_bonds_angles() {
    //Reset forces
    Kokkos::deep_copy(f,0);

    //compute bonded force
    compute_force_bonds();
    compute_force_angles();
    compute_force_dihedrals();
}

void particles_instance::compute_force_bonds() {
    Kokkos::parallel_for("bond-force",
        Kokkos::TeamPolicy<Tag_force_bonds>(h_bonds.extent(0), Kokkos::AUTO), *this);
}

KOKKOS_FUNCTION
void particles_instance::operator() (Tag_force_bonds, const member_type& team_member) const {
    const int i = team_member.league_rank();

    int atom1 = bonds(i).atom1 - 1;
    int atom2 = bonds(i).atom2 - 1;
    int type = bonds(i).type - 1;
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

    // Compute the force magnitude based on Hooke's law (F = -dV/dr)
    double dr = r - r0;
    double force_mag = -2.0 * k * dr / r;

    // Apply the forces to the atoms
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, 3), [&](const int& d) {
        double f_component = force_mag * (d == 0 ? dx : (d == 1 ? dy : dz));
        Kokkos::atomic_add(&f(atom1,d), -f_component);
        Kokkos::atomic_add(&f(atom2,d), f_component);
    });
}

void particles_instance::compute_force_angles() {
    Kokkos::parallel_for("angle-force", 
        Kokkos::RangePolicy<>(0, h_angles.extent(0)), *this);
}

KOKKOS_FUNCTION
void particles_instance::operator() (const int i) const {
    int atom1 = angles(i).atom1 - 1;
    int atom2 = angles(i).atom2 - 1;
    int atom3 = angles(i).atom3 - 1;
    int type = angles(i).type - 1;
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
}

void particles_instance::compute_force_dihedrals() {
    Kokkos::parallel_for("dihedral-force",
        Kokkos::TeamPolicy<Tag_force_dihedrals>(h_dihedrals.extent(0), Kokkos::AUTO), *this);
}

KOKKOS_FUNCTION
void particles_instance::operator() (Tag_force_dihedrals, const member_type& team_member) const {
    const int i = team_member.league_rank();

    int atom1 = dihedrals(i).atom1 - 1;
    int atom2 = dihedrals(i).atom2 - 1;
    int atom3 = dihedrals(i).atom3 - 1;
    int atom4 = dihedrals(i).atom4 - 1;
    int type = dihedrals(i).type - 1;

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
}

double particles_instance::potential_bonds_angles() {
    // add LJ-potential, since it doesnt make sense to compute bonds without it
    double bond_energy = potential_bonds();
    //printf("bond_E: %f ",bond_energy);
    double angle_energy = potential_angles();
    //printf("angle_energy: %f ",angle_energy);
    double dihedral_energy = potential_dihedrals();
    //printf("dihedral_energy: %f ",dihedral_energy);

    return bond_energy + angle_energy + dihedral_energy;
}

double particles_instance::potential_bonds() {
    double result = 0.0;
    Kokkos::parallel_reduce("bond-potential",
        Kokkos::TeamPolicy<Tag_potential_bonds>(h_bonds.extent(0), Kokkos::AUTO), *this, result);
    return result;
}

KOKKOS_FUNCTION
void particles_instance::operator() (Tag_potential_bonds, const member_type& team_member, double& V) const {
    const int i = team_member.league_rank();

    int atom1 = bonds(i).atom1-1;
    int atom2 = bonds(i).atom2-1;
    int type = bonds(i).type-1;
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

    Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
        V += potential;
    });
}

double particles_instance::potential_angles() {
    double result;
    Kokkos::parallel_reduce("angle-potential",
        Kokkos::TeamPolicy<Tag_potential_angles>(h_angles.extent(0), Kokkos::AUTO), *this, result);
    return result;
}

KOKKOS_FUNCTION
void particles_instance::operator() (Tag_potential_angles, const member_type& team_member, double& V) const {
    const int i = team_member.league_rank();

    int atom1 = angles(i).atom1 - 1;
    int atom2 = angles(i).atom2 - 1;
    int atom3 = angles(i).atom3 - 1;
    int type = angles(i).type -1;
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

    Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
        V += potential;
    });
}

double particles_instance::potential_dihedrals() {
    double result = 0.0;
    Kokkos::parallel_reduce("dihedral-potential",
        Kokkos::TeamPolicy<Tag_potential_dihedrals>(h_dihedrals.extent(0), Kokkos::AUTO), *this, result);
    return result;
}

KOKKOS_FUNCTION
void particles_instance::operator() (Tag_potential_dihedrals, const member_type& team_member, double& V) const {
    const int i = team_member.league_rank();

    int atom1 = dihedrals(i).atom1 - 1;
    int atom2 = dihedrals(i).atom2 - 1;
    int atom3 = dihedrals(i).atom3 - 1;
    int atom4 = dihedrals(i).atom4 - 1;
    int type = dihedrals(i).type - 1;

    double k1 = dihedralTypes(type).k1;
    double k2 = dihedralTypes(type).k2;
    double k3 = dihedralTypes(type).k3;
    double k4 = dihedralTypes(type).k4;
    //Kokkos::printf("k1: %f\n", k1);
    //Kokkos::printf("k2: %f\n", k2);
    //Kokkos::printf("k3: %f\n", k3);
    //Kokkos::printf("k4: %f\n", k4);

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
        dx -= round(dx / L[d]) * L[d];
        r12[d] = dx;

        dx = x3[d] - x2[d];
        dx -= round(dx / L[d]) * L[d];
        r23[d] = dx;

        dx = x4[d] - x3[d];
        dx -= round(dx / L[d]) * L[d];
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

    Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
        V += potential;
    });
}

void particles_instance::build_bondless_verlet_list() {
    build_verlet_list();
    Kokkos::deep_copy(h_verlet_list,verlet_list);
    Kokkos::deep_copy(h_neighbour_count,neighbour_count);
    /*printf("NEIGHBORS: %d \n",h_neighbour_count(3));
    printf("LIST: %d \n",h_verlet_list(3,0));
    printf("LIST: %d \n",h_verlet_list(3,1));
    printf("LIST: %d \n",h_verlet_list(3,2));
    printf("LIST: %d \n",h_verlet_list(3,3));
    printf("LIST: %d \n",h_verlet_list(3,4));
    printf("LIST: %d \n",h_verlet_list(3,5));
    printf("LIST: %d \n",h_verlet_list(3,6));
    printf("LIST: %d \n",h_verlet_list(3,7));
    printf("LIST: %d \n",h_verlet_list(3,8));
    printf("LIST: %d \n",h_verlet_list(3,9));
    printf("LIST: %d \n",h_verlet_list(3,10));*/
    // Remove bonded atoms from neighbour list
    Kokkos::parallel_for("verlet_remove_bonds",
        Kokkos::TeamPolicy<Tag_verlet_remove_bonds>(N, Kokkos::AUTO), *this);
    Kokkos::fence();
    Kokkos::deep_copy(h_verlet_list,verlet_list);
    Kokkos::deep_copy(h_neighbour_count,neighbour_count);
    /*printf("NEIGHBORS: %d \n",h_neighbour_count(3));
    printf("LIST: %d \n",h_verlet_list(3,0));
    printf("LIST: %d \n",h_verlet_list(3,1));
    printf("LIST: %d \n",h_verlet_list(3,2));
    printf("LIST: %d \n",h_verlet_list(3,3));
    printf("LIST: %d \n",h_verlet_list(3,4));
    printf("LIST: %d \n",h_verlet_list(3,5));
    printf("LIST: %d \n",h_verlet_list(3,6));
    printf("LIST: %d \n",h_verlet_list(3,7));
    printf("LIST: %d \n",h_verlet_list(3,8));
    printf("LIST: %d \n",h_verlet_list(3,9));
    printf("LIST: %d \n",h_verlet_list(3,10));*/
}

KOKKOS_FUNCTION
void particles_instance::operator()(Tag_verlet_remove_bonds, const member_type& teamMember) const {
    const int i = teamMember.league_rank();

    int initial_neighbour_count = neighbour_count(i);

    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, bonds.extent(0)), [=](const int b) {
        int atom1 = bonds(b).atom1 - 1;
        int atom2 = bonds(b).atom2 - 1;
        if (atom1 != i) return;
        // Search for atom2 in atom i's verlet list and remove it
        int n = neighbour_count(i);
        for (int k = 0; k < n; k++) {
            if (verlet_list(i, k) == atom2) {
                // Found the bonded atom, remove it by setting it to 0
                // we set it to 0 to first collect all atoms to be removed from the bond list, 
                // this avoids racing conditions. Since the verlet_list is directional (each bonds occurs
                // only once), atom "0" can never occur in the bond list and we can safely use the 0 as a placeholder
                verlet_list(i,k) = 0;
                
                // Decrement the neighbor count
                Kokkos::atomic_fetch_add(&neighbour_count(i),-1);
                break;
            }
        }
    });
    // Now we clean up the list on a single thread by finding rach 0
    // and then shifting all further values in the array "up".
    Kokkos::single(Kokkos::PerTeam(teamMember), [=]() {
        int n = initial_neighbour_count; // Start with the initial neighbor count
        int write_idx = 0; // Index to write the valid elements

        for (int read_idx = 0; read_idx < n; ++read_idx) {
            if (verlet_list(i, read_idx) != 0) {
                // Copy non-zero values to the write index
                verlet_list(i, write_idx) = verlet_list(i, read_idx);
                ++write_idx; // Increment the write index for the next valid element
            }
        }
    });


    // Remove atoms indirectly bonded to `i` (angle interactions)
    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, angles.extent(0)), [=](const int a) {
        int atom1 = angles(a).atom1 - 1;
        int atom2 = angles(a).atom2 - 1;
        int atom3 = angles(a).atom3 - 1;

        if (atom1 != i) return;

        // Remove atom2 from verlet list
        int n = neighbour_count(i);
        for (int k = 0; k < n; k++) {
            if (verlet_list(i, k) == atom2) {
                verlet_list(i, k) = 0;  // Mark for removal
                Kokkos::atomic_fetch_add(&neighbour_count(i), -1);
                break;
            }
        }

        // Remove atom3 from verlet list
        n = neighbour_count(i);
        for (int k = 0; k < n; k++) {
            if (verlet_list(i, k) == atom3) {
                verlet_list(i, k) = 0;  // Mark for removal
                Kokkos::atomic_fetch_add(&neighbour_count(i), -1);
                break;
            }
        }
    });

    // Clean up list after angle removal
    Kokkos::single(Kokkos::PerTeam(teamMember), [=]() {
        int n = initial_neighbour_count;
        int write_idx = 0;

        for (int read_idx = 0; read_idx < n; ++read_idx) {
            if (verlet_list(i, read_idx) != 0) {
                verlet_list(i, write_idx) = verlet_list(i, read_idx);
                ++write_idx;
            }
        }
    });

    // Remove atoms connected via dihedrals (1-4 interactions)
    /*Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, dihedrals.extent(0)), [=](const int d) {
        int atom1 = dihedrals(d).atom1 - 1;
        int atom2 = dihedrals(d).atom2 - 1;
        int atom3 = dihedrals(d).atom3 - 1;
        int atom4 = dihedrals(d).atom4 - 1;

        if (atom1 != i) return;

        // Remove atom2
        int n = neighbour_count(i);
        for (int k = 0; k < n; k++) {
            if (verlet_list(i, k) == atom2) {
                verlet_list(i, k) = 0;  // Mark for removal
                Kokkos::atomic_fetch_add(&neighbour_count(i), -1);
                break;
            }
        }

        // Remove atom3
        n = neighbour_count(i);
        for (int k = 0; k < n; k++) {
            if (verlet_list(i, k) == atom3) {
                verlet_list(i, k) = 0;  // Mark for removal
                Kokkos::atomic_fetch_add(&neighbour_count(i), -1);
                break;
            }
        }

        // Remove atom4
        n = neighbour_count(i);
        for (int k = 0; k < n; k++) {
            if (verlet_list(i, k) == atom4) {
                verlet_list(i, k) = 0;  // Mark for removal
                Kokkos::atomic_fetch_add(&neighbour_count(i), -1);
                break;
            }
        }
    });

    // Clean up list after dihedral removal
    Kokkos::single(Kokkos::PerTeam(teamMember), [=]() {
        int n = initial_neighbour_count;
        int write_idx = 0;

        for (int read_idx = 0; read_idx < n; ++read_idx) {
            if (verlet_list(i, read_idx) != 0) {
                verlet_list(i, write_idx) = verlet_list(i, read_idx);
                ++write_idx;
            }
        }
    });*/
}