#include "Umbrella_Sampling.hpp"
#include "particles.hpp"
#include <iomanip>
#include <set>
#include <stdexcept>

constexpr double density_conversion = 18.01528e24 / N_A;

Umbrella_Sampling::Umbrella_Sampling(const YAML::Node& config)
{
    if (!config["output_file"] || !config["output_every"]) {
        throw std::runtime_error("Umbrella sampling requires output_file and output_every.");
    }
    output_filename = config["output_file"].as<std::string>();
    output_every = config["output_every"].as<int>();
    if (output_filename.empty()) {
        throw std::runtime_error("Umbrella sampling output_file cannot be empty.");
    }
    if (output_every <= 0) {
        throw std::runtime_error("Umbrella sampling output_every must be positive.");
    }

    if (config["density"]) {
        if (!config["density"]["spring_constant"] || !config["density"]["center"]) {
            throw std::runtime_error("Umbrella sampling density requires spring_constant and center.");
        }
        density_enabled = true;
        density_spring_constant = config["density"]["spring_constant"].as<double>() * kcaltointernal * density_conversion * density_conversion;
        density_center = config["density"]["center"].as<double>() / density_conversion;
        if (density_spring_constant < 0.0) {
            throw std::runtime_error("Umbrella sampling density spring_constant cannot be negative.");
        }
    }

    if (config["Q6"]) {
        if (!config["Q6"]["spring_constant"] || !config["Q6"]["center"]) {
            throw std::runtime_error("Umbrella sampling Q6 requires spring_constant and center.");
        }
        Q6_enabled = true;
        Q6_spring_constant = config["Q6"]["spring_constant"].as<double>() * kcaltointernal;
        Q6_center = config["Q6"]["center"].as<double>();
        if (Q6_spring_constant < 0.0) {
            throw std::runtime_error("Umbrella sampling Q6 spring_constant cannot be negative.");
        }
    }

    if (!density_enabled && !Q6_enabled) {
        throw std::runtime_error("Umbrella sampling requires density or Q6.");
    }
}

void Umbrella_Sampling::init(const particles_instance& particles)
{
    std::ifstream existing_output(output_filename, std::ios::binary | std::ios::ate);
    bool write_header = !existing_output || existing_output.tellg() == 0;
    existing_output.close();
    output.open(output_filename, std::ios::app);
    if (!output) {
        throw std::runtime_error("Could not open umbrella sampling output file " + output_filename + ".");
    }
    if (write_header) {
        output << std::setprecision(12)
               << "# density_center(g/cm^3) " << density_center * density_conversion
               << " density_spring_constant(kcal/mol/(g/cm^3)^2) " << density_spring_constant / (kcaltointernal * density_conversion * density_conversion)
               << " Q6_center " << Q6_center
               << " Q6_spring_constant(kcal/mol) " << Q6_spring_constant / kcaltointernal << std::endl;
        output << "# step density(g/cm^3) Q6 bias_energy(kcal/mol)" << std::endl;
    }

    std::set<int> molecule_ids;
    for (std::size_t i = 0; i < particles.N; ++i) {
        molecule_ids.insert(particles.h_mol_id(i));
    }
    molecule_count = molecule_ids.size();
    cv_manager.init(particles);
    cv_manager.compute_CVs(particles);
    density = cv_manager.density;
    Q6 = cv_manager.Q6;
    trial_density = density;
    trial_Q6 = Q6;
    bias_energy = compute_bias_energy(density, Q6);
    trial_bias_energy = bias_energy;
}

void Umbrella_Sampling::evaluate_trial(const particles_instance& particles)
{
    cv_manager.compute_CVs(particles);
    trial_density = cv_manager.density;
    trial_Q6 = cv_manager.Q6;
    trial_bias_energy = compute_bias_energy(trial_density, trial_Q6);
}

void Umbrella_Sampling::evaluate_volume_trial(const particles_instance& particles)
{
    trial_density = double(molecule_count) / (particles.L[0] * particles.L[1] * particles.L[2]);
    trial_Q6 = Q6;
    trial_bias_energy = compute_bias_energy(trial_density, trial_Q6);
}

void Umbrella_Sampling::accept_trial()
{
    density = trial_density;
    Q6 = trial_Q6;
    bias_energy = trial_bias_energy;
}

void Umbrella_Sampling::reject_trial()
{
    trial_density = density;
    trial_Q6 = Q6;
    trial_bias_energy = bias_energy;
}

void Umbrella_Sampling::write_output(int step)
{
    if (step % output_every != 0) return;
    output << std::setprecision(12)
           << step << " "
           << density * density_conversion << " "
           << Q6 << " "
           << bias_energy / kcaltointernal << std::endl;
}

double Umbrella_Sampling::compute_bias_energy(double density_value, double Q6_value) const
{
    double energy = 0.0;
    if (density_enabled) {
        double difference = density_value - density_center;
        energy += 0.5 * density_spring_constant * difference * difference;
    }
    if (Q6_enabled) {
        double difference = Q6_value - Q6_center;
        energy += 0.5 * Q6_spring_constant * difference * difference;
    }
    return energy;
}
