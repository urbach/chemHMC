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
    std::vector<double> coordinates;
    std::vector<std::string> atom_names;
    std::vector<std::string> monomer_ids;
    std::vector<size_t> monomer_number_of_atoms;
    size_t n_monomers;
    bblock::System system;
};