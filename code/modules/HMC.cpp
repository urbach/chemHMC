#include "HMC.hpp"
#include "global.hpp"
#include <iostream>
#include <fstream>
#include "Input_reader.hpp"
#include "Parameters.hpp"
#include "Calc_Manager.hpp"
#include "particles.hpp"
#include <memory>
#include <cmath>

void HMC_class::init(int argc, char** argv, bool check_overwrite) {
    // Init main submodules
    params = new params_class();
    particles = new particles_instance();
    integrator = nullptr; // Specific integrator gets initialized by input reader
    calc_manager = new Calc_Manager();
    Input_reader input_reader = Input_reader(params, integrator, particles, calc_manager);
    // Create a std::shared_ptr from the raw pointer and pass it to the calc manager
    calc_manager->set_particles(std::shared_ptr<particles_instance>(particles));
    input_reader.parse_input(argc, argv);
    integrator->set_calc_manager(*calc_manager);
    // Initialize all Calc objects
    calc_manager->initialize();
    gen64.seed(params->seed);
    MD = input_reader.MD;
    UseNeighborList = input_reader.UseNeighborList;
    // Optional minimization
    doc = input_reader.doc;
    if (doc["minimization"]) calc_manager->minimize_energy(doc);
}

double HMC_class::gen_random() {
    return (((double)gen64() - gen64.min()) / (gen64.max() - gen64.min()));// random number from 0 to 1
};

void HMC_class::run() {

    Kokkos::Timer timer;
    double tokcal = 1.0/kcaltointernal;

    double Vi = calc_manager->compute_potential();
    printf("INITIAL V: %f \n", Vi*tokcal);

    double beta = integrator->particles->get_beta();
    calc_manager->compute_force();

    Kokkos::fence();
    // copy the configuration before the MD
    Kokkos::deep_copy(integrator->particles->h_x, integrator->particles->x);// h_x=x;
    for (int i = 1; i <= params->Ntrajectories; i++) {
        Kokkos::Timer timer_traj;
        // hb momenta
        integrator->particles->hb();
        double Ki = integrator->particles->compute_kinetic_E();
        // molecular dynamics
        integrator->integrate();

        // accept/reject
        double Vf = calc_manager->compute_potential();
        double Kf = integrator->particles->compute_kinetic_E();

        double dh = beta * (Kf + Vf - Ki - Vi);
        double exp_mdh = exp(-dh);

        if ((i % params->print_info_every == 0)) {
            printf("step %d: K = %.12g  V = %.12g exp_mdh = %.12g\n", i, Kf*tokcal, Vf*tokcal, exp_mdh);
        }
        Kokkos::fence();

        if (i < params->thermalization_steps) {
            Vi = Vf;
            Ki = Kf;
            Kokkos::deep_copy(integrator->particles->h_x, integrator->particles->x);// h_x=x;
        }
        else {
            double r = gen_random();// random number from 0 to 1
            if (r < exp_mdh) {
                acceptance++;
                Vi = Vf;
                Ki = Kf;
                Kokkos::deep_copy(integrator->particles->h_x, integrator->particles->x);// h_x=x;
            }
            else {
                Kokkos::deep_copy(integrator->particles->x, integrator->particles->h_x);
            }
            // save
            if ((i % params->save_every == 0)) {
                integrator->particles->print_xyz(*params, i, Ki, Vi);
            }
        }
    }
    printf("Acceptance: %g\n", acceptance / ((double)(params->Ntrajectories - params->thermalization_steps)));
    //printf("final step size: %f\n", integrator->dt);
    printf("time for HMC: %g  s\n", timer.seconds());
    calc_manager->print_timings();
    if (integrator->particles->neighbor_list_used) {
        printf("time for Neighbor list builds: %g s\n",particles->neighbor_list->time_list_build);
    }
}

void HMC_class::run_MD() {

    Kokkos::Timer timer;
    double tokcal = 1.0/kcaltointernal;

    double V_initial = calc_manager->compute_potential();
    printf("INITIAL V: %f \n", V_initial*tokcal);
    double Vi = V_initial;

    double beta = integrator->particles->get_beta();
    calc_manager->compute_force();

    Kokkos::fence();

    // hb momenta
    if (params->hb_momenta) {
        printf("Momenta initialized\n");
        integrator->particles->hb();
    }
    double K_initial = integrator->particles->compute_kinetic_E();
    printf("K_initial: %f\n",K_initial*tokcal);
    double Ki = K_initial;
    for (int i = 1; i <= params->Ntrajectories; i++) {
        // molecular dynamics
        integrator->integrate();
        if ((i % params->print_info_every == 0)) {
            double Vf = calc_manager->compute_potential();
            double Kf = integrator->particles->compute_kinetic_E();
            printf("trajectory = %d: deltaE = %.12g\n", i, fabs((Kf + Vf) - (K_initial + V_initial)));
        }
        Kokkos::fence();
        if ((i % params->save_every == 0)) {
            double Vf = calc_manager->compute_potential();
            double Kf = integrator->particles->compute_kinetic_E();
            Kokkos::deep_copy(integrator->particles->h_x, integrator->particles->x);
            integrator->particles->print_deltaE(*params, i, fabs((Kf + Vf) - (K_initial + V_initial)));
        }
    }
    double K_last = integrator->particles->compute_kinetic_E();
    double V_last = calc_manager->compute_potential();
    printf("time for MD: %g  s\n", timer.seconds());
    calc_manager->print_timings();
}