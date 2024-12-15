#ifndef ATOM_HPP
#define ATOM_HPP
#include "global.hpp"
#include <Kokkos_Core.hpp>

/**
 * @class atom_type
 * @brief Represents an atom with its elemental properties and force parameters.
 *
 * The atom_type class stores the elemental properties (label, mass, charge) and
 * force parameters (Lennard-Jones epsilon and sigma) of an atom. It provides a
 * constructor to initialize these properties.
 */
const int MAX_LABEL_LENGTH = 10;

class atom_type {

public:
    /**
     * @brief Elemental properties and force parameters of the atom.
     */
    char label[MAX_LABEL_LENGTH];   ///< The label of the atom (e.g. "C").
    double mass;            ///< The mass of the atom in atomic mass units (amu).
    double charge;          ///< The electric charge of the atom in elementary charges (e).
    int type_index;         ///< An index representing the type of atom (e.g., for use in a type array).
    // force parameters
    double LJ_epsilon;      ///< The Lennard-Jones parameter epsilon in Kelvin (K).
    double LJ_sigma;        ///< The Lennard-Jones parameter sigma in Angstrom (Å).

    //constructor
    KOKKOS_FUNCTION atom_type(const char* l, double m, double c, int i, double e, double s)
        : mass(m), charge(c), type_index(i), LJ_epsilon(e), LJ_sigma(s) {
        //strncpy(label, l, MAX_LABEL_LENGTH - 1);
        // Cant use strncpy on device

        // fill the list with null characters so it gets terminated properly when
        // the label has fewer characters than MAX_LABEL_LENGTH
        for (int j = 0; j < MAX_LABEL_LENGTH; ++j) {
            label[j] = '\0';
        }
        for (int j = 0; j < MAX_LABEL_LENGTH - 1 && l[j] != '\0'; ++j) {
            label[j] = l[j];
        }
        label[MAX_LABEL_LENGTH - 1] = '\0';
    }

    KOKKOS_FUNCTION atom_type()
        : mass(0), charge(0), type_index(0), LJ_epsilon(0), LJ_sigma(0) {
        label[0] = '\0';  // Initialize the label to an empty string
    }

    /*KOKKOS_FUNCTION atom_type(std::string label, double mass, double charge, int type_index, double LJ_epsilon, double LJ_sigma)
        : label(label), mass(mass), charge(charge), type_index(type_index), LJ_epsilon(LJ_epsilon), LJ_sigma(LJ_sigma) {}

    KOKKOS_FUNCTION atom_type() 
        : label(""), mass(0), charge(0), type_index(0), LJ_epsilon(0), LJ_sigma(0) {}*/
};

#endif