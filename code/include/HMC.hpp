#ifndef HMC_H
#define HMC_H
#include "read_infile.hpp"
#include "particles.hpp"
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
    std::string rng_host_state;
    params_class params;

    HMC_class() {};
    void save_host_rng_state();
    void init(int argc, char** argv, bool check_overwrite = true);

    void run();
    double gen_random();

    void measure();
};
#endif // !HMC_H
