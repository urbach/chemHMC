#ifndef HMC_H
#define HMC_H
#include "Parameters.hpp"
#include "yaml-cpp/yaml.h"
#include "particles.hpp"
#include "global.hpp"
#include "../modules/integrators/integrator.hpp"
#include "Calc_Manager.hpp"
#include <random>

class HMC_class {
public:
    integrator_type* integrator;
    int Ntrajectories;
    int thermalization_steps;
    int save_every;
    int print_info_every;
    int acceptance = 0;
    bool randomize_traj = false;
    YAML::Node doc;
    bool MD = false;
    // we need a random generator on the host for the accept/reject
    std::mt19937_64 gen64;
    params_class* params;
    particles_instance* particles;
    Calc_Manager* calc_manager;

    HMC_class() {};
    void init(int argc, char** argv, bool check_overwrite = true);

    void run();
    void run_MD();
    double gen_random();
};
#endif // !HMC_H
