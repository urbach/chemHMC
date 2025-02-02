#ifndef OMF2_HPP
#define OMF2_HPP

#include "integrator_type.hpp"

class OMF2 : public integrator_type {

public:
    const double lambda;
    const double oneminus2lambda;
    OMF2() = delete;
    OMF2(YAML::Node doc, params_class params);
    void integrate() override;

};

#endif