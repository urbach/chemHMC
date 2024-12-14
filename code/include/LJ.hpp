#ifndef LJ_HPP
#define LJ_HPP

#include "Calc.hpp"

class LJ : public Calc {
public:
    // Constructor and Destructor
    LJ() = default;
    ~LJ() override = default;

    // Override Calc methods
    void init() override;
    double potential() override;
    double force() override;

private:
    // Example parameters for Lennard-Jones
    double epsilon;
    double sigma;
};

#endif // LJ_HPP
