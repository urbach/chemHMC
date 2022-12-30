#include "HMC.hpp"
#include "read_infile.hpp"

void HMC_class::init(int argc, char** argv) {

    YAML::Node doc = read_params(argc, argv);

    if (doc["integrator"]) {
        std::string name = check_and_assign_value<std::string>(doc["integrator"], "name");
        if (name == "LEAP")
            integrator = new LEAP(doc);
        else {
            printf("no valid integrator name: ");
            std::cout << doc["integrator"].as<std::string>() << std::endl;
            exit(1);
        }
    }
    else {
        Kokkos::abort("no itegrator in input file");
    }

}

void HMC_class::run() {
    integrator->particles->hb();
    integrator->particles->printp();

    double V = integrator->particles->compute_potential();
    printf("the potential is: %f\n", V);
}