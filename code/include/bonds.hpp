#ifndef BONDS_HPP
#define BONDS_HPP

#include <Kokkos_Core.hpp>
#include "Calc.hpp"
#include "global.hpp"

struct BondType {
    int type;
    double k;    // Force constant
    double r0;   // Equilibrium distance
};

struct Bond {
    int id;
    int type;
    int atom1;
    int atom2;
};

struct AngleType {
    int type;
    double k;    // Force constant
    double theta0; // Equilibrium angle (in degrees)
};

struct Angle {
    int id;
    int type;
    int atom1;
    int atom2;
    int atom3;
};

struct DihedralType {
    int type;
    double k1;    // OPLS stye Force constants
    double k2;
    double k3;
    double k4;
};

struct Dihedral {
    int id;
    int type;
    int atom1;
    int atom2;
    int atom3;
    int atom4;
};

class Bonds : public Calc {
public:
    // these hold the actual bond data for the calculation
    Kokkos::View<int*[3]> bond_list;
    Kokkos::View<int*[3]>::HostMirror h_bond_list;
    Kokkos::View<double*[2]> bond_parameters;
    Kokkos::View<double*[2]>::HostMirror h_bond_parameters;

    // these hold the data on types
    Kokkos::View<Bond*> bonds;
    Kokkos::View<Bond*>::HostMirror h_bonds;
    Kokkos::View<BondType*> bondTypes;
    Kokkos::View<BondType*>::HostMirror h_bondTypes;
    Kokkos::View<AngleType*> angleTypes;
    Kokkos::View<AngleType*>::HostMirror h_angleTypes;
    Kokkos::View<Angle*> angles;
    Kokkos::View<Angle*>::HostMirror h_angles;
    Kokkos::View<DihedralType*> dihedralTypes;
    Kokkos::View<DihedralType*>::HostMirror h_dihedralTypes;
    Kokkos::View<Dihedral*> dihedrals;
    Kokkos::View<Dihedral*>::HostMirror h_dihedrals;

    void init(const particles_instance& particles) override;
    double potential(const particles_instance& particles) override;
    struct Tag_potential_bonds {};
    double potential_bonds(const particles_instance& particles);
    void force(const particles_instance& particles, type_f& f) override;
};

#endif