#ifndef HMC_H
#define HMC_H
#include "read_infile.hpp"
#include "particles.hpp"
#include "integrator.hpp"
#include <random>

class HMC_class {
public:
    integrator_type *integrator;
    int Ntrajectories;
    int acceptance;
    // we need a random generator on the host for the accept/reject
    std::mt19937_64 gen64;

    HMC_class(){};

    void init(int argc, char** argv);
       
    void run();
    double gen_random();

};
#endif // !HMC_H
