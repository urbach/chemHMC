#ifndef UMBRELLA_SAMPLING_H
#define UMBRELLA_SAMPLING_H

#include "CV_Manager.hpp"
#include "yaml-cpp/yaml.h"
#include <fstream>

class Umbrella_Sampling {
public:
    double density = 0.0;
    double Q6 = 0.0;
    double trial_density = 0.0;
    double trial_Q6 = 0.0;
    double bias_energy = 0.0;
    double trial_bias_energy = 0.0;
    bool density_enabled = false;
    bool Q6_enabled = false;
    double density_spring_constant = 0.0;
    double density_center = 0.0;
    double Q6_spring_constant = 0.0;
    double Q6_center = 0.0;
    std::string output_filename;
    int output_every = 0;

    Umbrella_Sampling() = default;
    Umbrella_Sampling(const YAML::Node& config);

    void init(const particles_instance& particles);
    void evaluate_trial(const particles_instance& particles);
    void evaluate_volume_trial(const particles_instance& particles);
    void accept_trial();
    void reject_trial();
    void write_output(int step);
private:
    double compute_bias_energy(double density_value, double Q6_value) const;
    std::size_t molecule_count = 0;
    CV_Manager cv_manager;
    std::ofstream output;
};

#endif
