#include "integrator.hpp"
#include "read_infile.hpp"


integrator_type::integrator_type(YAML::Node doc) {
    if (doc["particles"]) {
        std::string name = check_and_assign_value<std::string>(doc["particles"], "name");
        if (name == "identical_particles")
            particles = new identical_particles(doc);
        else {
            printf("no valid name for particles: ");
            std::cout << doc["particles"].as<std::string>() << std::endl;
            exit(1);
        }
        particles->InitX();
    }
    else {
        Kokkos::abort("no particles in input file");
    }
}

LEAP::LEAP(YAML::Node doc): integrator_type(doc) {


}