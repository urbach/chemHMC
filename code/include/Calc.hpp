#ifndef CALC_HPP
#define CALC_HPP
#include "global.hpp"

class particles_instance;

class Calc {
public:
    virtual void init(const particles_instance& particles) = 0;
    virtual double potential(const particles_instance& particles) = 0;
    virtual void force(const particles_instance& particles, type_f& f) = 0;
    virtual ~Calc() = default;
};
#endif