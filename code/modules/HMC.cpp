#include "HMC.hpp"
#include "read_infile.hpp"
#include <fstream>

void HMC_class::init(int argc, char** argv) {

    YAML::Node doc = read_params(argc, argv);
    params = params_class(doc);

    if (doc["integrator"]) {
        std::string name = check_and_assign_value<std::string>(doc["integrator"], "name");
        if (name == "LEAP")
            integrator = new LEAP(doc, params);
        else if (name == "OMF2")
            integrator = new OMF2(doc, params);
        else if (name == "OMF4")
            integrator = new OMF4(doc, params);
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
    thermalization_steps = check_and_assign_value<int>(doc, "thermalization_steps");
    std::cout << "thermalization_steps:" << thermalization_steps << std::endl;
    save_every = check_and_assign_value<int>(doc, "save_every");
    std::cout << "save_every:" << save_every << std::endl;
    int seed = check_and_assign_value<int>(doc, "seed");

    gen64.seed(seed);
    if (params.append == true) {
        std::cout << std::endl << "Loading rng host...\n";
        std::ifstream fin(params.rng_host_state);
        fin >> gen64;
    }

    acceptance = 0;
}

double HMC_class::gen_random() {
    return (((double)gen64() - gen64.min()) / (gen64.max() - gen64.min()));// random number from 0 to 1
};

void HMC_class::save_host_rng_state() {
    // save state
    std::cout << "Saving rng host...\n";
    {
        std::ofstream fout(params.rng_host_state);
        fout << gen64;
    }
}

void HMC_class::run() {

    Kokkos::Timer timer;
    double Vi = integrator->particles->compute_potential();

    double beta = integrator->particles->get_beta();

    Kokkos::fence();
    int first_traj = params.istart + 1;
    int last_traj = Ntrajectories + params.istart + 1;
    // copy the configuration before the MD
    Kokkos::deep_copy(integrator->particles->h_x, integrator->particles->x);// h_x=x;
    for (int i = first_traj; i < last_traj; i++) {
        Kokkos::Timer timer_traj;
        printf("Starting trajectory %d\n", i);
        // hb momenta
        integrator->particles->hb();
        double Ki = integrator->particles->compute_kinetic_E();
        printf("initial Action values betaS= %.12g   K= %.12g  V= %.12g\n", beta * (Vi + Ki), Ki, Vi);

        // molecular dynamics
        integrator->integrate();

        // accept/reject
        double Vf = integrator->particles->compute_potential();
        double Kf = integrator->particles->compute_kinetic_E();

        double dh = beta * (Kf + Vf - Ki - Vi);
        double exp_mdh = exp(-dh);
        printf("Action after the MD evolution betaS= %.12g   K= %.12g  V= %.12g beta= %.12g dh= %.12g exp_mdh= %.12g\n",
            beta * (Vf + Kf), Kf, Vf, beta, dh, exp_mdh);
        Kokkos::fence();


        if (i < thermalization_steps) {
            Vi = Vf;
            Ki = Kf;
            Kokkos::deep_copy(integrator->particles->h_x, integrator->particles->x);// h_x=x;
            printf("New configuration accepted during thermalization\n");
        }
        else {
            double r = gen_random();// random number from 0 to 1
            if (r < exp_mdh) {
                acceptance++;
                Vi = Vf;
                Ki = Kf;
                Kokkos::deep_copy(integrator->particles->h_x, integrator->particles->x);// h_x=x;
                printf("New configuration accepted\n");
            }
            else {
                Kokkos::deep_copy(integrator->particles->x, integrator->particles->h_x);
                printf("New configuration rejected\n");
            }
            // save
            if ((i % save_every == 0)) {
                printf("saving conf\n");
                integrator->particles->print_xyz(params, i, Ki, Vi);
                save_host_rng_state();
                integrator->particles->save_device_rng();
            }
        }
#ifdef DEBUG
        integrator->particles->printx();
#endif

        printf("time for trajectory: %g s\n\n", timer_traj.seconds());
    }
    printf("Acceptance: %g\n", acceptance / ((double)(Ntrajectories - thermalization_steps)));
    printf("time for HMC: %g  s\n", timer.seconds());


}


void HMC_class::measure() {
    auto& p = integrator->particles;
    FILE* file = NULL;
    file = fopen(params.nameout.c_str(), "r");
    if (file == NULL) {
        printf("error in opening file %s\n", params.nameout.c_str());
        Kokkos::abort("abort");
    }
    int confs = p->how_many_confs_xyz(file);
    printf("the input file contains %d configurations \n", confs);
    ////////////////////
    FILE* file_RDF = NULL;
    file_RDF = fopen(p->filename_RDF.c_str(), "w");
    if (file_RDF == NULL) {
        printf("error in opening file %s\n", p->filename_RDF.c_str());
        Kokkos::abort("abort");
    }
    p->write_header_RDF(file_RDF, confs);

    ///////////////////////
    for (int i = 0; i < confs; i++) {
        p->read_next_confs_xyz(file);
        p->compute_RDF();
        p->write_RDF(file_RDF, i);
    }
    fclose(file_RDF);
    fclose(file);
}