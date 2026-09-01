#include "HMC.hpp"
#include "global.hpp"
#include <iostream>
#include <fstream>
#include "Input_reader.hpp"
#include "Parameters.hpp"
#include "Calc_Manager.hpp"
#include "particles.hpp"
#include <memory>

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
    UseNeighborList = input_reader.UseNeighborList;
    // Optional minimization
    doc = input_reader.doc;
    { // Set simulation type
        std::string simulation_type_string = check_and_assign_value<std::string>(doc, "simulation_type");
        if (simulation_type_string == "MD") simulation_type = SimulationType::MD;
        if (simulation_type_string == "HMC") simulation_type = SimulationType::HMC;
        if (simulation_type_string == "VolumeMoveHMC") {
            simulation_type = SimulationType::VolumeMoveHMC;
            params->pressure = check_and_assign_value<double>(doc, "pressure");
            params->volume_step = check_and_assign_value<double>(doc, "max_volume_step");
        }
        if (simulation_type == SimulationType::None) Kokkos::abort("No simulation type selected! Aborting...");
    }
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

void HMC_class::run_VolumeMoveHMC() {

    Kokkos::Timer timer;
    double tokcal = 1.0/kcaltointernal;

    double P_atm = params->pressure;
    double atm_to_internal_pressure = 6.10e-9;
    double P_ext = P_atm * atm_to_internal_pressure;

    double max_delta_lnV = params->volume_step;

    double Vi = calc_manager->compute_potential();
    printf("INITIAL V: %f \n", Vi*tokcal);

    double beta = integrator->particles->get_beta();
    calc_manager->compute_force();

    Kokkos::fence();

    Kokkos::deep_copy(integrator->particles->h_x, integrator->particles->x);

    size_t volume_acceptance = 0;
    size_t volume_attempts = 0;

    // start NPT run 
    // every cycle attempts a HMC move and then a Volume move
    for (int i = 1; i <= params->Ntrajectories; i++) {
        Kokkos::Timer timer_traj;

        // HMC move
        integrator->particles->hb(); // set initial momenta
        double Ki = integrator->particles->compute_kinetic_E();
        integrator->integrate();

        double Vf = calc_manager->compute_potential();
        double Kf = integrator->particles->compute_kinetic_E();
        double dh = beta * (Kf + Vf - Ki - Vi);
        double exp_mdh = exp(-dh);
        Kokkos::fence();

        // accept/reject HMC move
        if (i < params->thermalization_steps) {
            Vi = Vf;
            Ki = Kf;
            Kokkos::deep_copy(integrator->particles->h_x, integrator->particles->x);
        }
        else {
            double r = gen_random();

            if (r < exp_mdh) {
                acceptance++;
                Vi = Vf;
                Ki = Kf;
                Kokkos::deep_copy(integrator->particles->h_x, integrator->particles->x);
            }
            else {
                Kokkos::deep_copy(integrator->particles->x, integrator->particles->h_x);
            }
        }

        // Attempt volume move
        if (i >= params->thermalization_steps) volume_attempts++;
        double L_old[3] = {
            integrator->particles->L[0],
            integrator->particles->L[1],
            integrator->particles->L[2]
        };

        double V_old = L_old[0] * L_old[1] * L_old[2];
        Kokkos::deep_copy(integrator->particles->h_x, integrator->particles->x);
        std::vector<double> x_old(3 * integrator->particles->N);
        for (size_t a = 0; a < integrator->particles->N; a++) {
            x_old[3*a + 0] = integrator->particles->h_x(a, 0);
            x_old[3*a + 1] = integrator->particles->h_x(a, 1);
            x_old[3*a + 2] = integrator->particles->h_x(a, 2);
        }

        // Since this is a constant-N run the number of mols never changes and we can just
        // set it to the initial value
        size_t n_mol = integrator->particles->number_of_molecules; 

        std::vector<double> com(3 * n_mol, 0.0);
        std::vector<double> mol_mass(n_mol, 0.0);
        std::vector<size_t> first_atom(n_mol, integrator->particles->N);

        for (size_t a = 0; a < integrator->particles->N; a++) {
            int mol = integrator->particles->h_mol_id(a);
            if (first_atom[mol] == integrator->particles->N) first_atom[mol] = a;
        }

        for (size_t a = 0; a < integrator->particles->N; a++) {
            int mol = integrator->particles->h_mol_id(a);
            int type = integrator->particles->h_id(a);
            double m = integrator->particles->h_atom_type_list(type).mass;

            double dx = integrator->particles->h_x(a, 0) - integrator->particles->h_x(first_atom[mol], 0);
            double dy = integrator->particles->h_x(a, 1) - integrator->particles->h_x(first_atom[mol], 1);
            double dz = integrator->particles->h_x(a, 2) - integrator->particles->h_x(first_atom[mol], 2);

            dx -= round(dx / L_old[0]) * L_old[0];
            dy -= round(dy / L_old[1]) * L_old[1];
            dz -= round(dz / L_old[2]) * L_old[2];

            com[3*mol + 0] += m * dx;
            com[3*mol + 1] += m * dy;
            com[3*mol + 2] += m * dz;
            mol_mass[mol] += m;
        }

        for (size_t mol = 0; mol < n_mol; mol++) {
            com[3*mol + 0] = integrator->particles->h_x(first_atom[mol], 0) + com[3*mol + 0] / mol_mass[mol];
            com[3*mol + 1] = integrator->particles->h_x(first_atom[mol], 1) + com[3*mol + 1] / mol_mass[mol];
            com[3*mol + 2] = integrator->particles->h_x(first_atom[mol], 2) + com[3*mol + 2] / mol_mass[mol];
        }

        double delta_lnV = max_delta_lnV * (2.0 * gen_random() - 1.0);
        double V_new = V_old * exp(delta_lnV);
        double scale = pow(V_new / V_old, 1.0 / 3.0);

        // set new boxlength parameters
        integrator->particles->L[0] = L_old[0] * scale;
        integrator->particles->L[1] = L_old[1] * scale;
        integrator->particles->L[2] = L_old[2] * scale;
        integrator->particles->inverse_L[0] = 1/integrator->particles->L[0];
        integrator->particles->inverse_L[1] = 1/integrator->particles->L[1];
        integrator->particles->inverse_L[2] = 1/integrator->particles->L[2];
        integrator->particles->inverse_halved_L[0] = 2*integrator->particles->inverse_L[0];
        integrator->particles->inverse_halved_L[1] = 2*integrator->particles->inverse_L[1];
        integrator->particles->inverse_halved_L[2] = 2*integrator->particles->inverse_L[2];

        for (size_t a = 0; a < integrator->particles->N; a++) {
            int mol = integrator->particles->h_mol_id(a);

            double dx = integrator->particles->h_x(a, 0) - com[3*mol + 0];
            double dy = integrator->particles->h_x(a, 1) - com[3*mol + 1];
            double dz = integrator->particles->h_x(a, 2) - com[3*mol + 2];

            integrator->particles->h_x(a, 0) = scale * com[3*mol + 0] + dx;
            integrator->particles->h_x(a, 1) = scale * com[3*mol + 1] + dy;
            integrator->particles->h_x(a, 2) = scale * com[3*mol + 2] + dz;
        }

        Kokkos::deep_copy(integrator->particles->x, integrator->particles->h_x);

        if (integrator->particles->neighbor_list_used) {
            integrator->particles->neighbor_list->build_verlet_list(*integrator->particles);
        }
        // accept/reject volume move
        double V_trial = calc_manager->compute_potential();
        double dH_vol = beta * (V_trial - Vi + P_ext * (V_new - V_old))
                      - double(n_mol + 1) * log(V_new / V_old);
        double exp_mdH_vol = exp(-dH_vol);
        if (gen_random() < exp_mdH_vol) { // accept
            if (i >= params->thermalization_steps) volume_acceptance++;
            Vi = V_trial;
            Kokkos::deep_copy(integrator->particles->h_x, integrator->particles->x);
        }
        else { // reject
            // restore old boxlength parameters
            integrator->particles->L[0] = L_old[0];
            integrator->particles->L[1] = L_old[1];
            integrator->particles->L[2] = L_old[2];
            integrator->particles->inverse_L[0] = 1/integrator->particles->L[0];
            integrator->particles->inverse_L[1] = 1/integrator->particles->L[1];
            integrator->particles->inverse_L[2] = 1/integrator->particles->L[2];
            integrator->particles->inverse_halved_L[0] = 2*integrator->particles->inverse_L[0];
            integrator->particles->inverse_halved_L[1] = 2*integrator->particles->inverse_L[1];
            integrator->particles->inverse_halved_L[2] = 2*integrator->particles->inverse_L[2];

            for (size_t a = 0; a < integrator->particles->N; a++) {
                integrator->particles->h_x(a, 0) = x_old[3*a + 0];
                integrator->particles->h_x(a, 1) = x_old[3*a + 1];
                integrator->particles->h_x(a, 2) = x_old[3*a + 2];
            }
            Kokkos::deep_copy(integrator->particles->x, integrator->particles->h_x);
            if (integrator->particles->neighbor_list_used) {
                integrator->particles->neighbor_list->build_verlet_list(*integrator->particles);
            }
        }

        // print info
        if ((i % params->print_info_every == 0)) {
            double K_now = integrator->particles->compute_kinetic_E();
            double V_now = calc_manager->compute_potential();
            double V_box = integrator->particles->L[0]
                         * integrator->particles->L[1]
                         * integrator->particles->L[2];

            printf("step %d: K = %.12g  V = %.12g Lx = %.12g Vol = %.12g exp_mdh = %.12g vol_acc = %.6g\n",
                   i,
                   K_now*tokcal,
                   V_now*tokcal,
                   integrator->particles->L[0],
                   V_box,
                   exp_mdh,
                   volume_attempts > 0 ? volume_acceptance / double(volume_attempts) : 0.0);
        }
        // print configuration
        if ((i % params->save_every == 0)) {
            double K_now = integrator->particles->compute_kinetic_E();
            double V_now = calc_manager->compute_potential();
            Kokkos::deep_copy(integrator->particles->h_x, integrator->particles->x);
            integrator->particles->print_xyz(*params, i, K_now, V_now);
        }
    }

    printf("Acceptance: %g\n", acceptance / ((double)(params->Ntrajectories - params->thermalization_steps)));
    printf("Volume acceptance: %g\n", volume_attempts > 0 ? volume_acceptance / double(volume_attempts) : 0.0);
    printf("time for HMC NPT: %g  s\n", timer.seconds());
    calc_manager->print_timings();

    if (integrator->particles->neighbor_list_used) {
        printf("time for Neighbor list builds: %g s\n",particles->neighbor_list->time_list_build);
    }
}

void HMC_class::run_MD() {

    Kokkos::Timer timer;
    double tokcal = 1.0/kcaltointernal;

    double Vi = calc_manager->compute_potential();
    printf("INITIAL V: %f \n", Vi*tokcal);

    double beta = integrator->particles->get_beta();
    calc_manager->compute_force();

    Kokkos::fence();

    // hb momenta
    if (params->hb_momenta) {
        printf("Momenta initialized\n");
        integrator->particles->hb();
    }
    double Ki = integrator->particles->compute_kinetic_E();
    printf("K_initial: %f\n",Ki*tokcal);
    for (int i = 1; i <= params->Ntrajectories; i++) {
        // molecular dynamics
        integrator->integrate();
        Kokkos::deep_copy(integrator->particles->h_x,integrator->particles->x);
        if ((i % params->print_info_every == 0)) {
            double Vf = calc_manager->compute_potential();
            double Kf = integrator->particles->compute_kinetic_E();
            printf("step %d: K = %.12g  V = %.12g H = %.12g \n", i, Kf*tokcal, Vf*tokcal, (Kf+Vf)*tokcal);
        }
        Kokkos::fence();
        if ((i % params->save_every == 0)) {
            double Vf = calc_manager->compute_potential();
            double Kf = integrator->particles->compute_kinetic_E();
            Kokkos::deep_copy(integrator->particles->h_x, integrator->particles->x);
            integrator->particles->print_xyz(*params, i, Kf, Vf);
        }
    }
    printf("time for MD: %g  s\n", timer.seconds());
    calc_manager->print_timings();
}