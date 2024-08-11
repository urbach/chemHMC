#ifndef HMC_H
#define HMC_H
#include "read_infile.hpp"
#include "particles_type.hpp"
#include "integrator.hpp"
#include <random>

class HMC_class {
public:
    integrator_type* integrator;
    int Ntrajectories;
    int thermalization_steps;
    int save_every;
    int acceptance;
    bool randomize_traj = false;
    // we need a random generator on the host for the accept/reject
    std::mt19937_64 gen64;
    params_class params;

    HMC_class() {};
    void init(int argc, char** argv, bool check_overwrite = true);

    void run();
    double gen_random();

    void measure();
};
#endif // !HMC_H
