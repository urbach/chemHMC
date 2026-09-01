#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "CV_Manager.hpp"
#include "Umbrella_Sampling.hpp"
#include "particles.hpp"

void initialize_particles(particles_instance& particles, const std::vector<double>& positions,
    const std::vector<int>& molecule_ids, const std::vector<double>& masses, const double box[3])
{
    particles.N = molecule_ids.size();
    particles.L[0] = box[0];
    particles.L[1] = box[1];
    particles.L[2] = box[2];
    particles.T = 300.0;
    particles.InitX();

    particles.atom_type_list = Kokkos::View<atom_type*>("atom_type_list", masses.size());
    particles.h_atom_type_list = Kokkos::create_mirror_view(particles.atom_type_list);
    for (std::size_t i = 0; i < masses.size(); ++i) {
        particles.h_atom_type_list(i) = atom_type("X", masses[i], 0.0, i, 0.0, 0.0);
    }
    Kokkos::deep_copy(particles.atom_type_list, particles.h_atom_type_list);

    for (std::size_t i = 0; i < molecule_ids.size(); ++i) {
        particles.h_mol_id(i) = molecule_ids[i];
        particles.h_id(i) = i % masses.size();
        particles.h_x(i, 0) = positions[3 * i + 0];
        particles.h_x(i, 1) = positions[3 * i + 1];
        particles.h_x(i, 2) = positions[3 * i + 2];
    }
    Kokkos::deep_copy(particles.x, particles.h_x);
    Kokkos::deep_copy(particles.id, particles.h_id);
    Kokkos::deep_copy(particles.mol_id, particles.h_mol_id);
}

bool reference_test()
{
    std::string path = std::string(CHEMHMC_SOURCE_DIR) + "/HMC_nosquish_release/examples/NPT_reaction_field/rst_b1.xyz";
    std::ifstream input(path);
    if (!input) {
        std::cerr << "Could not open " << path << std::endl;
        return false;
    }

    std::string line;
    std::getline(input, line);
    std::istringstream molecule_line(line);
    int molecule_count;
    molecule_line >> molecule_count;

    double box[3];
    input >> box[0] >> box[1] >> box[2];

    std::vector<double> positions(3 * molecule_count);
    std::vector<int> molecule_ids(molecule_count);
    for (int i = 0; i < molecule_count; ++i) {
        std::string name;
        double quaternion[4];
        input >> name >> positions[3 * i + 0] >> positions[3 * i + 1] >> positions[3 * i + 2]
              >> quaternion[0] >> quaternion[1] >> quaternion[2] >> quaternion[3];
        molecule_ids[i] = i;
    }

    particles_instance particles;
    initialize_particles(particles, positions, molecule_ids, {1.0}, box);

    CV_Manager manager;
    manager.init(particles);
    double potential = manager.potential(particles);

    bool passed = true;
    if (std::abs(manager.density - 0.033630204931875428) > 1.0e-14) {
        std::cerr << "Density mismatch: " << manager.density << std::endl;
        passed = false;
    }
    if (std::abs(manager.Q6 - 0.046181174050227210) > 1.0e-12) {
        std::cerr << "Q6 mismatch: " << manager.Q6 << std::endl;
        passed = false;
    }
    if (potential != 0.0) {
        std::cerr << "Potential mismatch: " << potential << std::endl;
        passed = false;
    }
    return passed;
}

bool periodic_com_test()
{
    double box[3] = {10.0, 10.0, 10.0};
    double oxygen_positions[5][3] = {
        {9.8, 5.0, 5.0},
        {0.8, 5.0, 5.0},
        {9.8, 6.0, 5.0},
        {9.8, 5.0, 6.0},
        {0.8, 6.0, 6.0}
    };

    std::vector<double> water_positions;
    std::vector<int> water_molecule_ids;
    std::vector<double> center_positions;
    std::vector<int> center_molecule_ids;
    for (int molecule = 0; molecule < 5; ++molecule) {
        double oxygen_x = oxygen_positions[molecule][0];
        double hydrogen_x = oxygen_x + 0.8;
        hydrogen_x -= floor(hydrogen_x / box[0]) * box[0];

        water_positions.push_back(oxygen_x);
        water_positions.push_back(oxygen_positions[molecule][1]);
        water_positions.push_back(oxygen_positions[molecule][2]);
        water_positions.push_back(hydrogen_x);
        water_positions.push_back(oxygen_positions[molecule][1]);
        water_positions.push_back(oxygen_positions[molecule][2]);
        water_positions.push_back(oxygen_x);
        water_positions.push_back(oxygen_positions[molecule][1] + 0.8);
        water_positions.push_back(oxygen_positions[molecule][2]);
        water_molecule_ids.push_back(molecule);
        water_molecule_ids.push_back(molecule);
        water_molecule_ids.push_back(molecule);

        double center_x = oxygen_x + 0.8 / 18.0;
        center_x -= floor(center_x / box[0]) * box[0];
        center_positions.push_back(center_x);
        center_positions.push_back(oxygen_positions[molecule][1] + 0.8 / 18.0);
        center_positions.push_back(oxygen_positions[molecule][2]);
        center_molecule_ids.push_back(molecule);
    }

    particles_instance water_particles;
    initialize_particles(water_particles, water_positions, water_molecule_ids, {16.0, 1.0, 1.0}, box);
    CV_Manager water_manager;
    water_manager.init(water_particles);
    water_manager.compute_CVs(water_particles);

    particles_instance center_particles;
    initialize_particles(center_particles, center_positions, center_molecule_ids, {1.0}, box);
    CV_Manager center_manager;
    center_manager.init(center_particles);
    center_manager.compute_CVs(center_particles);

    if (std::abs(water_manager.Q6 - center_manager.Q6) > 1.0e-12) {
        std::cerr << "Periodic COM Q6 mismatch: " << water_manager.Q6 << " " << center_manager.Q6 << std::endl;
        return false;
    }
    return true;
}

bool umbrella_energy_test(particles_instance& particles)
{
    bool passed = true;
    std::string density_output = "/tmp/chemHMC_test_density_cv.out";
    std::string Q6_output = "/tmp/chemHMC_test_Q6_cv.out";
    std::string combined_output = "/tmp/chemHMC_test_combined_cv.out";
    std::remove(density_output.c_str());
    std::remove(Q6_output.c_str());
    std::remove(combined_output.c_str());

    {
        YAML::Node config = YAML::Load(
            "output_file: /tmp/chemHMC_test_density_cv.out\n"
            "output_every: 1\n"
            "density:\n"
            "  spring_constant: 20.0\n"
            "  center: 0.04\n");
        Umbrella_Sampling umbrella(config);
        umbrella.init(particles);
        double difference = umbrella.density - umbrella.density_center;
        double expected = 0.5 * kB * particles.T * umbrella.density_spring_constant * difference * difference;
        if (std::abs(umbrella.bias_energy - expected) > 1.0e-14) passed = false;
    }

    {
        YAML::Node config = YAML::Load(
            "output_file: /tmp/chemHMC_test_Q6_cv.out\n"
            "output_every: 1\n"
            "Q6:\n"
            "  spring_constant: 30.0\n"
            "  center: 0.05\n");
        Umbrella_Sampling umbrella(config);
        umbrella.init(particles);
        double difference = umbrella.Q6 - umbrella.Q6_center;
        double expected = 0.5 * kB * particles.T * umbrella.Q6_spring_constant * difference * difference;
        if (std::abs(umbrella.bias_energy - expected) > 1.0e-14) passed = false;
    }

    {
        YAML::Node config = YAML::Load(
            "output_file: /tmp/chemHMC_test_combined_cv.out\n"
            "output_every: 2\n"
            "density:\n"
            "  spring_constant: 20.0\n"
            "  center: 0.04\n"
            "Q6:\n"
            "  spring_constant: 30.0\n"
            "  center: 0.05\n");
        Umbrella_Sampling umbrella(config);
        umbrella.init(particles);

        double density_difference = umbrella.density - umbrella.density_center;
        double Q6_difference = umbrella.Q6 - umbrella.Q6_center;
        double expected = 0.5 * kB * particles.T
                        * (umbrella.density_spring_constant * density_difference * density_difference
                        + umbrella.Q6_spring_constant * Q6_difference * Q6_difference);
        if (std::abs(umbrella.bias_energy - expected) > 1.0e-14) passed = false;

        double density = umbrella.density;
        double Q6 = umbrella.Q6;
        double bias_energy = umbrella.bias_energy;
        double old_box[3] = {particles.L[0], particles.L[1], particles.L[2]};
        particles.L[0] *= 1.1;
        particles.L[1] *= 1.1;
        particles.L[2] *= 1.1;

        umbrella.evaluate_volume_trial(particles);
        double trial_density = umbrella.trial_density;
        double trial_bias_energy = umbrella.trial_bias_energy;
        if (umbrella.trial_Q6 != Q6) passed = false;
        umbrella.reject_trial();
        if (umbrella.trial_density != density || umbrella.trial_Q6 != Q6 || umbrella.trial_bias_energy != bias_energy) passed = false;

        umbrella.evaluate_volume_trial(particles);
        umbrella.accept_trial();
        if (umbrella.density != trial_density || umbrella.Q6 != Q6 || umbrella.bias_energy != trial_bias_energy) passed = false;

        umbrella.write_output(1);
        umbrella.write_output(2);
        particles.L[0] = old_box[0];
        particles.L[1] = old_box[1];
        particles.L[2] = old_box[2];
    }

    std::ifstream output(combined_output);
    std::string line;
    int line_count = 0;
    int field_count = 0;
    while (std::getline(output, line)) {
        line_count++;
        std::istringstream values(line);
        double value;
        while (values >> value) field_count++;
    }
    if (line_count != 1 || field_count != 8) passed = false;

    std::remove(density_output.c_str());
    std::remove(Q6_output.c_str());
    std::remove(combined_output.c_str());
    return passed;
}

bool umbrella_input_test()
{
    bool passed = true;
    try {
        YAML::Node config = YAML::Load(
            "density:\n"
            "  spring_constant: 1.0\n"
            "  center: 0.04\n");
        Umbrella_Sampling umbrella(config);
        passed = false;
    }
    catch (const std::exception&) {
    }

    try {
        YAML::Node config = YAML::Load(
            "output_file: /tmp/chemHMC_test_invalid_cv.out\n"
            "output_every: 1\n"
            "density:\n"
            "  spring_constant: 1.0\n");
        Umbrella_Sampling umbrella(config);
        passed = false;
    }
    catch (const std::exception&) {
    }

    try {
        YAML::Node config = YAML::Load(
            "output_file: /tmp/chemHMC_test_invalid_cv.out\n"
            "output_every: 1\n"
            "Q6:\n"
            "  spring_constant: -1.0\n"
            "  center: 0.05\n");
        Umbrella_Sampling umbrella(config);
        passed = false;
    }
    catch (const std::exception&) {
    }

    try {
        YAML::Node config = YAML::Load(
            "output_file: /tmp/chemHMC_test_invalid_cv.out\n"
            "output_every: 0\n"
            "Q6:\n"
            "  spring_constant: 1.0\n"
            "  center: 0.05\n");
        Umbrella_Sampling umbrella(config);
        passed = false;
    }
    catch (const std::exception&) {
    }
    return passed;
}

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    bool passed;
    {
        std::string path = std::string(CHEMHMC_SOURCE_DIR) + "/HMC_nosquish_release/examples/NPT_reaction_field/rst_b1.xyz";
        std::ifstream input(path);
        std::string line;
        std::getline(input, line);
        std::istringstream molecule_line(line);
        int molecule_count;
        molecule_line >> molecule_count;
        double box[3];
        input >> box[0] >> box[1] >> box[2];
        std::vector<double> positions(3 * molecule_count);
        std::vector<int> molecule_ids(molecule_count);
        for (int i = 0; i < molecule_count; ++i) {
            std::string name;
            double quaternion[4];
            input >> name >> positions[3 * i + 0] >> positions[3 * i + 1] >> positions[3 * i + 2]
                  >> quaternion[0] >> quaternion[1] >> quaternion[2] >> quaternion[3];
            molecule_ids[i] = i;
        }
        particles_instance particles;
        initialize_particles(particles, positions, molecule_ids, {1.0}, box);
        passed = reference_test() && periodic_com_test() && umbrella_energy_test(particles) && umbrella_input_test();
    }
    Kokkos::finalize();
    return passed ? 0 : 1;
}
