#ifndef GENERAL_HPP
#define GENERAL_HPP

#include "integrator_type.hpp"
#include <vector>

// General splitting schemes for H = A + B with
//   A = kinetic energy   -> Q(a dt): position step (drift), coefficients a
//   B = potential energy -> P(b dt): momentum step (kick), coefficients b
// This is the convention of the scheme files in argon_testsystem/Data/Schemes:
// standard_parameters_a -> a, standard_parameters_b -> b (after expanding the symmetric halves).
// Every kick needs one force evaluation, so a step costs q = cycles force evaluations.

// AB ordering, drift first:
//   Q(a[0] dt) P(b[0] dt) Q(a[1] dt) ... P(b[q-1] dt) Q(a[q] dt)
//   a: q+1 drift coefficients, b: q kick coefficients
class GENERAL_AB : public integrator_type {
public:
    int cycles;
    std::vector<double> a;
    std::vector<double> b;

    GENERAL_AB(YAML::Node doc, params_class params);
    void integrate() override;
};

// BA ordering, kick first:
//   P(b[0] dt) Q(a[0] dt) P(b[1] dt) ... Q(a[q-1] dt) P(b[q] dt)
//   a: q drift coefficients, b: q+1 kick coefficients
class GENERAL_BA : public integrator_type {
public:
    int cycles;
    std::vector<double> a;
    std::vector<double> b;

    GENERAL_BA(YAML::Node doc, params_class params);
    void integrate() override;
};

#endif
