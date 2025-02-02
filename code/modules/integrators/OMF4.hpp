#ifndef OMF4_HPP
#define OMF4_HPP

#include "integrator_type.hpp"

class OMF4 : public integrator_type {

public:
    const double rho;
    const double theta;
    const double vartheta;
    const double lambda;
    const double dtau;
    const double eps[10];
    OMF4() = delete;
    OMF4(YAML::Node doc, params_class params);
    void integrate() override;

};

#endif