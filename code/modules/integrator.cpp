#include "integrator.hpp"
#include "read_infile.hpp"
#include "identical_particles.hpp"


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

    steps = check_and_assign_value<int>(doc["integrator"], "steps");
    dt = check_and_assign_value<double>(doc["integrator"], "dt");
    std::cout << "steps: " << steps << std::endl;
    std::cout << "dt: " << dt << std::endl;

}

LEAP::LEAP(YAML::Node doc): integrator_type(doc) {

}



void LEAP::integrate() {

    // initial half-step for the  momenta
    particles->update_momenta(dt / 2.);
    // first full step for the position
    particles->update_positions(dt);
    // nsteps-1 full steps
    for (size_t i = 0; i < steps - 1; i++) {
        particles->update_momenta(dt);
        particles->update_positions(dt);
    }
    // final half-step for the momenta
    particles->update_momenta(dt / 2.);
}


//////////////////////////////////////////////////////////////////////////////
// OMF2
//////////////////////////////////////////////////////////////////////////////
OMF2::OMF2(YAML::Node doc): integrator_type(doc), lambda(0.1938), oneminus2lambda(1. - 2. * lambda) {
}



void OMF2::integrate() {

    // initial half-step for the  momenta
    particles->update_momenta(lambda * dt);

    // nsteps-1 full steps
    for (size_t i = 0; i < steps - 1; i++) {
        particles->update_positions(dt / 2.);
        particles->update_momenta(oneminus2lambda * dt);
        particles->update_positions(dt / 2.);
        particles->update_momenta(2. * lambda * dt);
    }
    // final step
    particles->update_positions(dt / 2.);
    particles->update_momenta(oneminus2lambda * dt);
    particles->update_positions(dt / 2.);
    particles->update_momenta( lambda * dt);
}