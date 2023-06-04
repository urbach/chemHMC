#include "HMC.hpp"
#include "read_infile.hpp"

void HMC_class::init(int argc, char** argv) {

    YAML::Node doc = read_params(argc, argv);

    if (doc["integrator"]) {
        std::string name = check_and_assign_value<std::string>(doc["integrator"], "name");
        if (name == "LEAP")
            integrator = new LEAP(doc);
        else if (name == "OMF2")
            integrator = new OMF2(doc);
        else {
            printf("no valid integrator name: ");
            std::cout << doc["integrator"]["name"].as<std::string>() << std::endl;
            exit(1); // TODO: call Kokkos::abort
        }
    }
    else {
        Kokkos::abort("no itegrator in input file");
    }

    Ntrajectories = check_and_assign_value<int>(doc, "Ntrajectories");
    std::cout << "Ntrajectories:" << Ntrajectories << std::endl;
    int seed = check_and_assign_value<int>(doc, "seed");
    gen64.seed(seed);
    acceptance = 0;
}

double HMC_class::gen_random(){
    return (((double)gen64() - gen64.min()) / (gen64.max() - gen64.min()));// random number from 0 to 1
};

void HMC_class::run() {

    double Vi = integrator->particles->compute_potential();
#ifdef DEBUG
    printf("the potential is: %f\n", Vi);
    Kokkos::fence();
#endif // DEBUG
    // copy the configuration before the MD
    Kokkos::deep_copy(integrator->particles->h_x, integrator->particles->x);// h_x=x;
    for (int i = 0; i < Ntrajectories; i++) {
        
        integrator->particles->hb();
#ifdef DEBUG
        integrator->particles->printx();
        integrator->particles->printp();
#endif //DEBUG
        integrator->integrate();

        // accept/reject
        double Vf = integrator->particles->compute_potential();
        double r = gen_random();// random number from 0 to 1
#ifdef DEBUG
        printf("the potential after the MD evolution is: %f\n", Vf);
#endif //DEBUG
        Kokkos::fence();
        if (r < exp(-(Vf - Vi))) {
            acceptance++;
            Vi = Vf;
            Kokkos::deep_copy(integrator->particles->h_x, integrator->particles->x);// h_x=x;
#ifdef DEBUG
            printf("accepting the configuration\n");
#endif //DEBUG
        }
        else {
            Kokkos::deep_copy(integrator->particles->x, integrator->particles->h_x);
#ifdef DEBUG
            printf("rejecting the configuration\n");
#endif //DEBUG
        }
#ifdef DEBUG
        integrator->particles->printx();
        integrator->particles->printp();
#endif //DEBUG
    }


}