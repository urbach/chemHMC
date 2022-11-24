#include "HMC.hpp"

void HMC_class::init(int argc, char** argv) {
    params = new params_class();
    YAML::Node doc = params->read_params(argc, argv);

    if (doc["particles"]["name"].as<std::string>() == "identical_particles")
        particles = new identical_particles(doc["particles"], *params);
    else {
        printf("no valid name for particles: ");
        std::cout << doc["particles"].as<std::string>() << std::endl;
        exit(1);
    }
    
    particles->InitX();
    
}

void HMC_class::run() {
    particles->hb();
    particles->printp();

    double V=particles->compute_potential();
    printf("the potential is: %f\n",V);
}