#include "HMC.h"

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