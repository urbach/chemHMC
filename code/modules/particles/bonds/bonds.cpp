#include "global.hpp"
#include "atom.hpp"
#include "particles.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

void particles_instance::read_bonds_angles(const std::string& filename) {
    // parser for lammps style datafiles
    std::ifstream infile(filename);
    std::string line;

    int numBonds = 0, numAngles = 0, numBondTypes = 0, numAngleTypes = 0;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string keyword;
        int count;
        
        iss >> count >> keyword;

        if (keyword == "bonds") {
            numBonds = count;
        } else if (keyword == "angles") {
            numAngles = count;
        } else if (keyword == "BondTypes") {
            numBondTypes = count;
        } else if (keyword == "AngleTypes") {
            numAngleTypes = count;
        }

        if (numBonds > 0 && numAngles > 0 && numBondTypes > 0 && numAngleTypes > 0) {
            break;
        }
    }

    bonds = Kokkos::View<Bond*>("bonds", numBonds);
    angles = Kokkos::View<Angle*>("angles", numAngles);
    bondTypes = Kokkos::View<BondType*>("bondTypes", numBondTypes);
    angleTypes = Kokkos::View<AngleType*>("angleTypes", numAngleTypes);

    h_bonds = Kokkos::create_mirror_view(bonds);
    h_angles = Kokkos::create_mirror_view(angles);
    h_bondTypes = Kokkos::create_mirror_view(bondTypes);
    h_angleTypes = Kokkos::create_mirror_view(angleTypes);

    bool inBondSection = false, inAngleSection = false, inBondTypeSection = false, inAngleTypeSection = false;

    int bondIndex = 0, angleIndex = 0, bondTypeIndex = 0, angleTypeIndex = 0;

    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);

        // Read Bond Types section
        if (line.find("Bond Types") != std::string::npos) {
            inBondTypeSection = true;
            inAngleTypeSection = false;
            inBondSection = false;
            inAngleSection = false;
            continue;
        }

        // Read Angle Types section
        if (line.find("Angle Types") != std::string::npos) {
            inAngleTypeSection = true;
            inBondTypeSection = false;
            inBondSection = false;
            inAngleSection = false;
            continue;
        }

        // Read Bonds section
        if (line.find("Bonds") != std::string::npos) {
            inBondSection = true;
            inAngleSection = false;
            inBondTypeSection = false;
            inAngleTypeSection = false;
            continue;
        }

        // Read Angles section
        if (line.find("Angles") != std::string::npos) {
            inAngleSection = true;
            inBondSection = false;
            inBondTypeSection = false;
            inAngleTypeSection = false;
            continue;
        }

        // Parse bond type data if in Bond Types section
        if (inBondTypeSection && bondTypeIndex < numBondTypes) {
            int type;
            double k, r0;
            iss >> type >> k >> r0;
            h_bondTypes(bondTypeIndex).type = type;
            h_bondTypes(bondTypeIndex).k = k;
            h_bondTypes(bondTypeIndex).r0 = r0;
            bondTypeIndex++;
        }

        // Parse angle type data if in Angle Types section
        if (inAngleTypeSection && angleTypeIndex < numAngleTypes) {
            int type;
            double k, theta0;
            iss >> type >> k >> theta0;
            h_angleTypes(angleTypeIndex).type = type;
            h_angleTypes(angleTypeIndex).k = k;
            h_angleTypes(angleTypeIndex).theta0 = theta0 * M_PI/180.0;
            angleTypeIndex++;
        }

        // Parse bond data if in Bonds section
        if (inBondSection && bondIndex < numBonds) {
            int id, type, atom1, atom2;
            iss >> id >> type >> atom1 >> atom2;
            h_bonds(bondIndex).id = id;
            h_bonds(bondIndex).type = type;
            h_bonds(bondIndex).atom1 = atom1;
            h_bonds(bondIndex).atom2 = atom2;
            bondIndex++;
        }

        // Parse angle data if in Angles section
        if (inAngleSection && angleIndex < numAngles) {
            int id, type, atom1, atom2, atom3;
            iss >> id >> type >> atom1 >> atom2 >> atom3;
            h_angles(angleIndex).id = id;
            h_angles(angleIndex).type = type;
            h_angles(angleIndex).atom1 = atom1;
            h_angles(angleIndex).atom2 = atom2;
            h_angles(angleIndex).atom3 = atom3;
            angleIndex++;
        }
    }

    // Copy data to device
    Kokkos::deep_copy(bonds, h_bonds);
    Kokkos::deep_copy(angles, h_angles);
    Kokkos::deep_copy(bondTypes, h_bondTypes);
    Kokkos::deep_copy(angleTypes, h_angleTypes);

    // Output the counts and confirm the data read
    std::cout << "Number of Bonds: " << numBonds << std::endl;
    std::cout << "Number of Angles: " << numAngles << std::endl;
    std::cout << "Actual Bonds Read: " << bondIndex << std::endl;
    std::cout << "Actual Angles Read: " << angleIndex << std::endl;
    std::cout << "Number of Bond Types: " << numBondTypes << std::endl;
    std::cout << "Number of Angle Types: " << numAngleTypes << std::endl;

    for (int i = 0; i < bonds.extent(0); i++) {
        printf("bond number: %d \n", i);
        printf("atom1: %d atom2: %d k: %f r0: %f\n",h_bonds(i).atom1,h_bonds(i).atom2,h_bondTypes(h_bonds(i).type-1).k,h_bondTypes(h_bonds(i).type-1).r0);
    }
    for (int i = 0; i < angles.extent(0); i++) {
        printf("bond number: %d \n", i);
        printf("atom1: %d atom2: %d atom3: %d k: %f theta0: %f\n",h_angles(i).atom1,h_angles(i).atom2,h_angles(i).atom3,h_angleTypes(h_angles(i).type-1).k,h_angleTypes(h_angles(i).type-1).theta0);
    }
}

void particles_instance::compute_force_bonds_angles() {
    //Reset forces
    Kokkos::deep_copy(f,0);

    //compute LJ-force
    compute_force_verlet_list();

    //compute bonded force
    compute_force_bonds();
    compute_force_angles();
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

    // Calculate cosine of the angle
    double c = (delx1 * delx2 + dely1 * dely2 + delz1 * delz2) / (r1 * r2);
    c = fmin(fmax(c, -1.0), 1.0);  // Clamp c to the range [-1, 1]

    // Compute the angle deviation from the equilibrium angle
    double dtheta = acos(c) - theta0;

    // Force magnitude: simplified to be physically accurate
    double s = sqrt(1.0 - c * c);
    double a = -2.0 * k * dtheta / (s + 1e-8);

    // Force components
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

double particles_instance::potential_bonds_angles() {
    // add LJ-potential, since it doesnt make sense to compute bonds without it
    double result = potential_verlet_list();
    
    result += potential_bonds();
    result += potential_angles();
    return result;
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

    int atom1 = bonds(i).atom1 - 1;
    int atom2 = bonds(i).atom2 - 1;
    int type = bonds(i).type - 1;
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
    double potential = k*dr*dr;

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
    double potential = k * dtheta * dtheta;

    Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
        V += potential;
    });
}

void particles_instance::build_bondless_verlet_list() {
    build_verlet_list();
    Kokkos::deep_copy(h_verlet_list,verlet_list);
    Kokkos::deep_copy(h_neighbour_count,neighbour_count);
    // Remove bonded atoms from neighbour list
    Kokkos::parallel_for("verlet_remove_bonds",
        Kokkos::TeamPolicy<Tag_verlet_remove_bonds>(N, Kokkos::AUTO), *this);
    Kokkos::deep_copy(h_verlet_list,verlet_list);
    Kokkos::deep_copy(h_neighbour_count,neighbour_count);
}

KOKKOS_FUNCTION
void particles_instance::operator()(Tag_verlet_remove_bonds, const member_type& teamMember) const {
    const int i = teamMember.league_rank();
    //remove atoms from neighbour list of i that are directly bonded to i
    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, bonds.extent(0)), [=](const int b) {
        int atom1 = bonds(b).atom1 - 1;
        int atom2 = bonds(b).atom2 - 1;

        if (atom1 != i) return;
        //Kokkos::printf("atom1: %d atom2: %d \n", atom1, atom2);
        // Search for atom2 in atom i's verlet list and remove it
        int n = neighbour_count(i);
        for (int k = 0; k < n; k++) {
            if (verlet_list(i, k) == atom2) {
                // Found the bonded atom, remove it by shifting the remaining elements
                for (int l = k; l < n - 1; l++) {
                    verlet_list(i, l) = verlet_list(i, l + 1);
                }
                // Decrement the neighbor count
                Kokkos::atomic_fetch_add(&neighbour_count(i),-1);
                break;
            }
        }
    });

    //remove atoms from neighbour list of i that are indirectly bonded to i (angle interactions)
    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, angles.extent(0)), [=](const int b) {
        int atom1 = angles(b).atom1 - 1;
        int atom2 = angles(b).atom2 - 1;
        int atom3 = angles(b).atom3 - 1;


        if (atom1 != i) return;
        //Kokkos::printf("atom1: %d atom2: %d \n", atom1, atom2);
        // Search for atom2 in atom i's verlet list and remove it
        int n = neighbour_count(i);
        for (int k = 0; k < n; k++) {
            if (verlet_list(i, k) == atom2) {
                // Found the bonded atom, remove it by shifting the remaining elements
                for (int l = k; l < n - 1; l++) {
                    verlet_list(i, l) = verlet_list(i, l + 1);
                }
                // Decrement the neighbor count
                Kokkos::atomic_fetch_add(&neighbour_count(i),-1);
                break;
            }
        }
        // Search for atom3 in atom i's verlet list and remove it
        n = neighbour_count(i);
        for (int k = 0; k < n; k++) {
            if (verlet_list(i, k) == atom3) {
                // Found the bonded atom, remove it by shifting the remaining elements
                for (int l = k; l < n - 1; l++) {
                    verlet_list(i, l) = verlet_list(i, l + 1);
                }
                // Decrement the neighbor count
                Kokkos::atomic_fetch_add(&neighbour_count(i),-1);
                break;
            }
        }
    });
}