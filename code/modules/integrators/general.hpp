#ifndef GENERAL_HPP
#define GENERAL_HPP

#include "integrator_type.hpp"
#include <vector>

class GENERAL : public integrator_type {
public:
    int cycles;
    std::vector<double> a;
    std::vector<double> b;

    GENERAL(YAML::Node doc, params_class params);
    void integrate() override;
};

#endif