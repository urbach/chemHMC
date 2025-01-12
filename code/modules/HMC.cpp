#include "HMC.hpp"
#include "global.hpp"
#include <iostream>
#include <fstream>
#include "read_infile.hpp"
#include "Input_reader.hpp"
#include "Parameters.hpp"
#include "Calc_Manager.hpp"
#include "particles_type.hpp"
#include "particles.hpp"
#include "LJ.hpp"
#include <memory>

void HMC_class::init(int argc, char** argv, bool check_overwrite) {
    doc = read_params(argc, argv);
    params_o = params_class();
    params = &params_o;
    acceptance = 0;
    Input_reader input_reader = Input_reader(params, integrator, particles);
    input_reader.parse_input(argc, argv);
    gen64.seed(params->seed);
}

double HMC_class::gen_random() {
    return (((double)gen64() - gen64.min()) / (gen64.max() - gen64.min()));// random number from 0 to 1
};

void HMC_class::optimize_stepsize() {
    std::vector<double> step_sizes;
    double travel_distance;
    double average_acceptance;
    double current_best_travel_distance = 0.0;
    double current_best_dt = 0.0;
    double best_dt_acceptance;
    double Vi = integrator->particles->compute_potential();
    double beta = integrator->particles->get_beta();

    type_x x_copy = type_x("x_copy", integrator->particles->N);
    Kokkos::deep_copy(x_copy, integrator->particles->h_x);

    double start_dt = check_and_assign_value<double>(doc["optimize_stepsize"], "start");
    double stepsize_dt = check_and_assign_value<double>(doc["optimize_stepsize"], "stepsize");
    double stop_dt = check_and_assign_value<double>(doc["optimize_stepsize"], "stop");
    int N_traj = check_and_assign_value<int>(doc["optimize_stepsize"], "Ntesttrajectories");
    int N_steps = ((stop_dt - start_dt) / stepsize_dt)+1;
    integrator->dt = start_dt;

    // Check if we should print to a file
    std::string output_file;
    bool print_to_file = false;
    if (doc["optimize_stepsize"]["print_to_file"]) {
        output_file = doc["optimize_stepsize"]["print_to_file"].as<std::string>();
        print_to_file = true;
    }

    // File handling
    std::ofstream file;
    if (print_to_file) {
        file.open(output_file, std::ios::app); // Open file in append mode
        if (!file) {
            std::cerr << "Error opening file: " << output_file << std::endl;
            return;
        }
        // Write header
        file << "### Optimization Run ###\n";
        file << "dt\tacceptance\ttravel_distance\n";
    }

    for (int j = 1; j <= N_steps; j++) {
        // run test trajectories
        for (int i = 1; i < N_traj; i++) {
            Kokkos::Timer timer_traj;
            // hb momenta
            integrator->particles->hb();
            double Ki = integrator->particles->compute_kinetic_E();

            if (integrator->particles->algorithm == "verlet_list") {
                integrator->particles->build_verlet_list();
            }
            if (integrator->particles->algorithm == "bonds_angles") integrator->particles->build_bondless_verlet_list();
            if (integrator->particles->algorithm == "opls") integrator->particles->build_bondless_verlet_list();
            integrator->integrate();

            // accept/reject
            double Vf = integrator->particles->compute_potential();
            double Kf = integrator->particles->compute_kinetic_E();

            double dh = beta * (Kf + Vf - Ki - Vi);
            double exp_mdh = exp(-dh);
            Kokkos::fence();
            
            double r = gen_random(); // random number from 0 to 1
            if (r < exp_mdh) {
                acceptance++;
                Vi = Vf;
                Ki = Kf;
                Kokkos::deep_copy(integrator->particles->h_x, integrator->particles->x); // h_x=x;
            } else {
                Kokkos::deep_copy(integrator->particles->x, integrator->particles->h_x);
            }
        }

        // calculate travel distance and save it
        average_acceptance = ((double)acceptance) / ((double)(N_traj));
        acceptance = 0.0;

        travel_distance = integrator->dt * integrator->steps * average_acceptance;
        printf("dt: %f\t", integrator->dt);
        printf("acc: %f\t", average_acceptance);
        printf("travel distance: %f\n", travel_distance);
        if (travel_distance > current_best_travel_distance) {
            current_best_travel_distance = travel_distance;
            current_best_dt = integrator->dt;
            best_dt_acceptance = average_acceptance;
        }
        // Write to file if enabled
        if (print_to_file) {
            file << integrator->dt << "\t" << average_acceptance << "\t" << travel_distance << "\n";
        }

        // increment integrator
        integrator->dt += stepsize_dt;

        // reset system to initial configuration
        Kokkos::deep_copy(integrator->particles->x, x_copy);
        Kokkos::deep_copy(integrator->particles->h_x, x_copy);
    }

    printf("\n\n DONE \n\n\n");
    printf("optimized dt: %f \n", current_best_dt);
    printf("acceptance: %f \n", best_dt_acceptance);
    printf("travel_distance: %f \n", current_best_travel_distance);
    integrator->dt = current_best_dt;

    if (print_to_file) {
        // Write summary
        file << "\nOptimized dt: " << current_best_dt << "\n";
        file << "Acceptance: " << best_dt_acceptance << "\n";
        file << "Travel Distance: " << current_best_travel_distance << "\n\n";
        file.close(); // Close the file
    }
}

void HMC_class::run() {

    Kokkos::Timer timer;
    std::vector<double> acceptance_vec;
    double tokcal = 1.0/kcaltointernal;
    // perform a loose energy minimization to remove energy hotspots
    if (doc["minimization"]) {
        printf("STARTING MINIMIZATION \n");
        integrator->particles->minimize_energy(doc);
    }
    
    if (integrator->particles->algorithm == "verlet_list") integrator->particles->build_verlet_list();
    if (integrator->particles->algorithm == "bonds_angles") integrator->particles->build_bondless_verlet_list();
    if (integrator->particles->algorithm == "opls") integrator->particles->build_bondless_verlet_list();
    
    if (doc["optimize_stepsize"]) {
        optimize_stepsize();
        printf("USED DT: %f \n", integrator->dt);
    }
    
    double Vi = integrator->particles->compute_potential();
    printf("INITIAL V: %f \n", Vi*tokcal);

    double beta = integrator->particles->get_beta();

    Kokkos::fence();
    // copy the configuration before the MD
    Kokkos::deep_copy(integrator->particles->h_x, integrator->particles->x);// h_x=x;
    for (int i = 0; i < params->Ntrajectories; i++) {
        Kokkos::Timer timer_traj;
        // hb momenta
        integrator->particles->hb();
        double Ki = integrator->particles->compute_kinetic_E();
        // molecular dynamics
        if (integrator->particles->algorithm == "verlet_list") {
            integrator->particles->build_verlet_list();
        }
        if (integrator->particles->algorithm == "bonds_angles") integrator->particles->build_bondless_verlet_list();
        if (integrator->particles->algorithm == "opls") integrator->particles->build_bondless_verlet_list();
        integrator->integrate();

        // accept/reject
        double Vf = integrator->particles->compute_potential();
        double Kf = integrator->particles->compute_kinetic_E();

        double dh = beta * (Kf + Vf - Ki - Vi);
        double exp_mdh = exp(-dh);
        if ((i % params->print_info_every == 0)) {
            printf("step %d: K = %.12g  V = %.12g \n", i, Kf*tokcal, Vf*tokcal);
        }
        Kokkos::fence();


        if (i < params->thermalization_steps) {
            Vi = Vf;
            Ki = Kf;
            Kokkos::deep_copy(integrator->particles->h_x, integrator->particles->x);// h_x=x;
        }
        else {
            double r = gen_random();// random number from 0 to 1
            //acceptance_vec.push_back(exp_mdh);
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
            // adjust the stepsize based the current average acceptance
            if (doc["adjust_step_size"]) {
                if ((acceptance / ((double)(i - params->thermalization_steps))) > 0.66) {
                    integrator->dt *= 1.01;
                } else if ((acceptance / ((double)(i - params->thermalization_steps))) < 0.64) {
                    integrator->dt *= 0.99;
                }
            }
        }
#ifdef DEBUG
        integrator->particles->printx();
#endif
    }
    printf("Acceptance: %g\n", acceptance / ((double)(Ntrajectories - thermalization_steps)));
    printf("final step size: %f\n", integrator->dt);
    printf("time for HMC: %g  s\n", timer.seconds());
}

void HMC_class::run2() {

    Kokkos::Timer timer;
    double tokcal = 1.0/kcaltointernal;

    // Create a Calc_Manager instance
    Calc_Manager calc_manager;
    
    // Create a std::shared_ptr from the raw pointer and pass it to manager.set_particles
    calc_manager.set_particles(std::shared_ptr<particles_instance>(particles));

    // Create an LJ object and add it to the manager
    std::shared_ptr<Calc> ljCalc = std::make_shared<LJ>();
    calc_manager.addCalc(ljCalc);

    // Initialize all Calc objects
    calc_manager.initialize();

    integrator->set_calc_manager(calc_manager);

    double Vi = calc_manager.compute_potential();
    printf("INITIAL V: %f \n", Vi*tokcal);

    double beta = integrator->particles->get_beta();
    calc_manager.compute_force();
    //std::cout << "Lennard-Jones Force: " << force << std::endl;
    integrator->set_calc_manager(calc_manager);

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
        double Vf = calc_manager.compute_potential();
        double Kf = integrator->particles->compute_kinetic_E();

        double dh = beta * (Kf + Vf - Ki - Vi);
        double exp_mdh = exp(-dh);

        if ((i % params->print_info_every == 0)) {
            printf("step %d: K = %.12g  V = %.12g \n", i, Kf*tokcal, Vf*tokcal);
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
    printf("final step size: %f\n", integrator->dt);
    printf("time for HMC: %g  s\n", timer.seconds());
}