#include "MBX.hpp"

#include <vector>
#include <cstring>

std::vector<size_t> build_atom_to_monomer_map_h2o(const particles_instance& particles)
{
    std::vector<size_t> atom_to_monomer(particles.N);

    std::vector<size_t> oxygens;
    std::vector<size_t> hydrogens;

    for (size_t i = 0; i < particles.N; i++) {
        const char* label = particles.h_atom_type_list(particles.h_id(i)).label;

        if (std::strcmp(label, "O") == 0) oxygens.push_back(i);
        if (std::strcmp(label, "H") == 0) hydrogens.push_back(i);
    }

    std::vector<int> used(particles.N, 0);
    size_t monomer_id = 0;

    for (size_t oi = 0; oi < oxygens.size(); oi++) {
        size_t o = oxygens[oi];

        double ox = particles.h_x(o,0);
        double oy = particles.h_x(o,1);
        double oz = particles.h_x(o,2);

        size_t h1 = 0;
        size_t h2 = 0;
        double d1 = 1.0e100;
        double d2 = 1.0e100;

        for (size_t hi = 0; hi < hydrogens.size(); hi++) {
            size_t h = hydrogens[hi];
            if (used[h]) continue;

            double dx = ox - particles.h_x(h,0);
            double dy = oy - particles.h_x(h,1);
            double dz = oz - particles.h_x(h,2);
            double r2 = dx*dx + dy*dy + dz*dz;

            if (r2 < d1) {
                d2 = d1;
                h2 = h1;
                d1 = r2;
                h1 = h;
            } else if (r2 < d2) {
                d2 = r2;
                h2 = h;
            }
        }

        atom_to_monomer[o] = monomer_id;
        atom_to_monomer[h1] = monomer_id;
        atom_to_monomer[h2] = monomer_id;

        used[h1] = 1;
        used[h2] = 1;

        monomer_id++;
    }

    return atom_to_monomer;
}

void MBX::init(const particles_instance& particles) 
{
    system = bblock::System();
    size_t n_monomers = particles.num_monomers();

}