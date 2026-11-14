#ifndef DEEPMD_HPP
#define DEEPMD_HPP

#include "Calc.hpp"
#include "yaml-cpp/yaml.h"
#include "particles.hpp"
#include "deepmd/deepmd.hpp"
#include <string>
#include <vector>

class DeePMD : public Calc {
public:
    DeePMD(const YAML::Node& config);
    ~DeePMD() override = default;

    void init(const particles_instance& particles) override;
    double potential(const particles_instance& particles) override;
    void force(const particles_instance& particles, type_f& f) override;
private:
    double evaluate(const particles_instance& particles);

    deepmd::hpp::DeepPot dp;
    std::string model_file;
    int gpu_rank = 0;
    std::vector<double> coordinates;
    std::vector<int> atom_types;
    std::vector<double> box;
    std::vector<double> forces;
    // DeePMD interface automatically computes the virial and needs a
    // vector for it, but this is unused for now. 
    std::vector<double> virial; 

    Kokkos::View<double**, Kokkos::HostSpace> x_deepmd;
    Kokkos::View<double**, Kokkos::HostSpace> h_f;
};

#endif
