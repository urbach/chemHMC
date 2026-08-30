#ifndef MBX_HPP
#define MBX_HPP

#include <atom.hpp>
#include <iostream>
#include <iomanip>
#include "yaml-cpp/yaml.h"
#include "particles.hpp"
//MBX
#include "bblock/system.h"

class MBX : public Calc {
public:
    MBX() = default;
    MBX(const YAML::Node& config);
    ~MBX() override = default;

    void init(const particles_instance& particles) override;
    double potential(const particles_instance& particles) override;
    void force(const particles_instance& particles, type_f& f) override;
private:
    struct WaterMolecule {
        std::size_t oxygen;
        std::size_t hydrogen1;
        std::size_t hydrogen2;
    };

    void build_molecule_map(const particles_instance& particles);
    void update_box(const particles_instance& particles);
    void write_nrg_file(const particles_instance& particles, const std::string& filename);
    std::vector<double> coordinates_to_vector(const particles_instance& particles) const;

    std::vector<bblock::System> systems;
    std::vector<WaterMolecule> water_molecules;
    std::vector<std::size_t> mbx_to_particle;
    std::string json_filename = "mbx.json"; // json input default
    double box_length[3] = {0.0, 0.0, 0.0};
    bool box_initialized = false;

    Kokkos::View<double**, Kokkos::HostSpace> x_mbx;
    Kokkos::View<double**, Kokkos::HostSpace> h_f;
};

#endif