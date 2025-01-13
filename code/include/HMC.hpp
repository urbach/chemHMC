#ifndef HMC_H
#define HMC_H
#include "read_infile.hpp"
#include "yaml-cpp/yaml.h"
#include "particles_type.hpp"
#include "particles.hpp"
#include "global.hpp"
#include "integrator.hpp"
#include <random>

class HMC_class {
public:
    integrator_type* integrator;
    int Ntrajectories;
    int thermalization_steps;
    int save_every;
    int print_info_every;
    int acceptance;
    bool randomize_traj = false;
    YAML::Node doc;
    // we need a random generator on the host for the accept/reject
    std::mt19937_64 gen64;
    params_class* params;
    particles_instance* particles;

    HMC_class() {};
    void init(int argc, char** argv, bool check_overwrite = true);

    void run();
    void run2();
    void optimize_stepsize();
    double gen_random();
};
#endif // !HMC_H
