void compute_average_acceptance(int Ntrajectories, int thermalization_steps, std::vector<double> acceptance_vec) {
    int number_of_steps = ((double)(params->Ntrajectories - params->thermalization_steps));
    double sum = 0;
    for (size_t i = 0; i < number_of_steps; ++i)
    {
        sum += acceptance_vec[i]; 
    }
    printf("<e(-H)> = %f", sum/number_of_steps);
}

int particles_instance::how_many_confs_xyz(FILE* file) {

    int lines = 0;
    char c;

    /* count the newline characters */
    while ((c = fgetc(file)) != EOF) {
        if (c == '\n')
            lines++;
    }
    if (lines % (N + 2) != 0) {
        printf("error: xyz file contains %d lines\n", lines);
        printf("       the number of lines mus be a multiple of N+2=%d\n", N + 2);
        Kokkos::abort("abort");
    }
    int confs = lines / (N + 2);
    printf("confs in input configuration file %d\n", confs);
    rewind(file);
    return confs;
}

void particles_instance::read_next_confs_xyz(FILE* file) {
    int count = 0;
    char id[1000];
    char c;
    while ((c = fgetc(file)) != EOF) {
        if (c == '\n') {
            for (int i = 0;i < 11;i++) c = fgetc(file);
            int tmp;
            count += fscanf(file, " %d", &tmp);
            break;
        }
    }
    while ((c = fgetc(file)) != EOF) { if (c == '\n') break; }
    for (int i = 0; i < N;i++) {
        count += fscanf(file, "%s   %lf   %lf  %lf\n", id, &h_x(i, 0), &h_x(i, 1), &h_x(i, 2));
        // printf("%s   %lf   %lf  %lf\n", id, h_x(i, 0), h_x(i, 1), h_x(i, 2));
    }
    if (name_xyz.compare(id) != 0) {
        printf("name in the xyz file: %s  do not mach the name in the input file: %s\n", id, name_xyz.c_str());
        Kokkos::abort("abort");
    }
    // printf("%d  %d\n", count, N);
    if (count != N * 4 + 1) { Kokkos::abort("error in reading the file"); }
    Kokkos::deep_copy(x, h_x);
    // printx();
}

 void particles_type::save_device_rng() {
    // rand_pool.return_rng_state(hs);
    // FILE* f;
    // f = fopen(params.rng_device_state.c_str(), "w+");
    // printf("Saving rnd device...  %d %d\n",N,padding);
    // int i=fwrite(&hs(0, 0), sizeof(uint64_t), N * padding, f);
    // fclose(f);
}
void particles_type::load_device_rng() {
    // FILE* f;
    // f = fopen(params.rng_device_state.c_str(), "r");
    // int i=fread(&hs(0, 0), sizeof(uint64_t), N * padding, f);
    // if (i!=N*padding) Kokkos::abort("invalid rng_device_state file\n");
    // rand_pool.load_rng_state(hs);
    // fclose(f);
} 

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
                //integrator->particles->build_verlet_list();
            }
            //if (integrator->particles->algorithm == "bonds_angles") integrator->particles->build_bondless_verlet_list();
            //if (integrator->particles->algorithm == "opls") integrator->particles->build_bondless_verlet_list();
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