#ifndef CALC_HPP
#define CALC_HPP
#include "global.hpp"

class particles_instance;

class Calc {
public:
    double time_potential = 0.0;
    double time_force = 0.0;
    std::string name;

    virtual void init(const particles_instance& particles) = 0;
    virtual double potential(const particles_instance& particles) = 0;
    virtual void force(const particles_instance& particles, type_f& f) = 0;
    void print_timings() {
    printf("time for %s potential: %g  s\n", name.c_str(), time_potential);
    printf("time for %s force: %g  s\n", name.c_str(), time_force);
    }
    virtual ~Calc() = default;
};
#endif