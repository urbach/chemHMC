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