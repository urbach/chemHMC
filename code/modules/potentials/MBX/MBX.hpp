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
    ~MBX() override = default;

    void init(const particles_instance& particles) override;
    double potential(const particles_instance& particles) override;
    void force(const particles_instance& particles, type_f& f) override;
private:
    void write_nrg_file(const particles_instance& particles, const std::string& filename);
    std::vector<double> coordinates_to_vector(const particles_instance& particles) const;

    std::vector<bblock::System> systems;
    std::vector<std::size_t> mbx_to_particle;

    Kokkos::View<double**, Kokkos::HostSpace> x_mbx;
    Kokkos::View<double**, Kokkos::HostSpace> h_f;
};

#endif