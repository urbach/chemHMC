#ifndef ATOM_HPP
#define ATOM_HPP


/**
 * @class atom_type
 * @brief Represents an atom with its elemental properties and force parameters.
 *
 * The atom_type class stores the elemental properties (label, mass, charge) and
 * force parameters (Lennard-Jones epsilon and sigma) of an atom. It provides a
 * constructor to initialize these properties.
 */
class atom_type {

public:
    /**
     * @brief Elemental properties and force parameters of the atom.
     */
    std::string label;           ///< The label of the atom (e.g. "C").
    double mass;            ///< The mass of the atom in atomic mass units (amu).
    double charge;          ///< The electric charge of the atom in elementary charges (e).
    int type_index;         ///< An index representing the type of atom (e.g., for use in a type array).
    // force parameters
    double LJ_epsilon;      ///< The Lennard-Jones parameter epsilon in Kelvin (K).
    double LJ_sigma;        ///< The Lennard-Jones parameter sigma in Angstrom (Å).

    //constructor
    atom_type(std::string label, double mass, double charge, int type_index, double LJ_epsilon, double LJ_sigma)
        : label(label), mass(mass), charge(charge), type_index(type_index), LJ_epsilon(LJ_epsilon), LJ_sigma(LJ_sigma) {} 

    atom_type() : label(""), mass(0), charge(0), type_index(0), LJ_epsilon(0), LJ_sigma(0) {}
};

#endif