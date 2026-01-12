#include <atom.hpp>
#include <iostream>
#include <iomanip>
#include "yaml-cpp/yaml.h"
#include "particles.hpp"
#include <torch/script.h>

class MLIAP : public Calc {
public:
    MLIAP() = default;
    ~MLIAP() override = default;

    std::string model_path = "model.pt";
    std::string device_str = "cuda";

    void init(const particles_instance& particles) override;
    double potential(const particles_instance& particles) override;
    void force(const particles_instance& particles, type_f& f) override;

private:
    torch::Device device_{"cuda"};
    torch::jit::script::Module module_;
    int64_t N_at = 0;

    double L[dim_space];
    double inverse_L[dim_space];
    double inverse_halved_L[dim_space];
    double cutoff_squared = 0.0;
};