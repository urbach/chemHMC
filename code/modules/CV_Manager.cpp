#include "CV_Manager.hpp"
#include "particles.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <map>
#include <stdexcept>

void CV_Manager::init(const particles_instance& particles)
{
    name = "CV_Manager";
    density = 0.0;
    Q6 = 0.0;
    x_cv = Kokkos::View<double**, Kokkos::HostSpace>("x_cv", particles.N, 3);

    std::map<int, std::vector<std::size_t>> molecules;
    for (std::size_t i = 0; i < particles.N; ++i) {
        int mol_id = particles.h_mol_id(i);
        molecules[mol_id].push_back(i);
    }

    molecule_atoms.clear();
    for (std::map<int, std::vector<std::size_t>>::const_iterator it = molecules.begin(); it != molecules.end(); ++it) {
        molecule_atoms.push_back(it->second);
    }
    molecular_centers.resize(3 * molecule_atoms.size());
}

double CV_Manager::potential(const particles_instance& particles)
{
    compute_CVs(particles);
    return 0.0;
}

void CV_Manager::force(const particles_instance& particles, type_f& f)
{
    return;
}

void CV_Manager::compute_CVs(const particles_instance& particles)
{
    Kokkos::deep_copy(x_cv, particles.x);
    compute_molecular_centers(particles);
    build_Q6_neighbor_list(particles);
    compute_Q6();
    density = double(molecule_atoms.size()) / (particles.L[0] * particles.L[1] * particles.L[2]);
}

void CV_Manager::compute_molecular_centers(const particles_instance& particles)
{
    for (std::size_t mol = 0; mol < molecule_atoms.size(); ++mol) {
        std::size_t first_atom = molecule_atoms[mol][0];
        double mol_mass = 0.0;
        double relative_com[3] = {0.0, 0.0, 0.0};

        for (std::size_t atom : molecule_atoms[mol]) {
            int type = particles.h_id(atom);
            double mass = particles.h_atom_type_list(type).mass;
            mol_mass += mass;

            for (std::size_t dir = 0; dir < 3; ++dir) {
                double displacement = x_cv(atom, dir) - x_cv(first_atom, dir);
                displacement -= round(displacement / particles.L[dir]) * particles.L[dir];
                relative_com[dir] += mass * displacement;
            }
        }

        for (std::size_t dir = 0; dir < 3; ++dir) {
            molecular_centers[3 * mol + dir] = x_cv(first_atom, dir) + relative_com[dir] / mol_mass;
        }
    }
}

void CV_Manager::build_Q6_neighbor_list(const particles_instance& particles)
{
    Q6_neighbors.clear();
    Q6_neighbors.resize(molecule_atoms.size());

    for (std::size_t i = 0; i + 1 < molecule_atoms.size(); ++i) {
        for (std::size_t j = i + 1; j < molecule_atoms.size(); ++j) {
            double dx = molecular_centers[3 * j + 0] - molecular_centers[3 * i + 0];
            double dy = molecular_centers[3 * j + 1] - molecular_centers[3 * i + 1];
            double dz = molecular_centers[3 * j + 2] - molecular_centers[3 * i + 2];

            dx -= round(dx / particles.L[0]) * particles.L[0];
            dy -= round(dy / particles.L[1]) * particles.L[1];
            dz -= round(dz / particles.L[2]) * particles.L[2];

            double distance = sqrt(dx * dx + dy * dy + dz * dz);
            if (distance < 5.0) {
                Q6Neighbor neighbor_i = {j, {dx, dy, dz}, distance};
                Q6Neighbor neighbor_j = {i, {-dx, -dy, -dz}, distance};
                Q6_neighbors[i].push_back(neighbor_i);
                Q6_neighbors[j].push_back(neighbor_j);
            }
        }
    }

    for (std::size_t mol = 0; mol < Q6_neighbors.size(); ++mol) {
        if (Q6_neighbors[mol].size() < 4) {
            throw std::runtime_error("Molecule id " + std::to_string(mol) + " has fewer than four Q6 neighbors.");
        }

        std::stable_sort(Q6_neighbors[mol].begin(), Q6_neighbors[mol].end(),
            [](const Q6Neighbor& a, const Q6Neighbor& b) {
                return a.distance < b.distance;
            });
        Q6_neighbors[mol].resize(4);
    }
}

void CV_Manager::compute_Q6()
{
    double pi = acos(-1.0);
    double pre[7];
    pre[6] = 1.0 / 64.0 * sqrt(3003.0 / pi);
    pre[5] = 3.0 / 32.0 * sqrt(1001.0 / pi);
    pre[4] = 3.0 / 32.0 * sqrt(91.0 / (2.0 * pi));
    pre[3] = 1.0 / 32.0 * sqrt(1365.0 / pi);
    pre[2] = 1.0 / 64.0 * sqrt(1365.0 / pi);
    pre[1] = 1.0 / 16.0 * sqrt(273.0 / (2.0 * pi));
    pre[0] = 1.0 / 32.0 * sqrt(13.0 / pi);

    std::complex<double> global_Q6[13];
    for (int m = 0; m < 13; ++m) {
        global_Q6[m] = std::complex<double>(0.0, 0.0);
    }

    for (std::size_t mol = 0; mol < Q6_neighbors.size(); ++mol) {
        std::complex<double> molecule_Q6[13];
        for (int m = 0; m < 13; ++m) {
            molecule_Q6[m] = std::complex<double>(0.0, 0.0);
        }

        for (std::size_t neighbor = 0; neighbor < Q6_neighbors[mol].size(); ++neighbor) {
            double distance = Q6_neighbors[mol][neighbor].distance;
            double cost = Q6_neighbors[mol][neighbor].displacement[2] / distance;
            double sint = sqrt(1.0 - cost * cost);
            double cosp;
            double sinp;

            if (std::abs(sint) < 1.0e-9) {
                cosp = 1.0;
                sinp = 0.0;
            }
            else {
                cosp = Q6_neighbors[mol][neighbor].displacement[0] / (distance * sint);
                sinp = Q6_neighbors[mol][neighbor].displacement[1] / (distance * sint);
            }

            double post[7];
            post[6] = sint * sint * sint * sint * sint * sint;
            post[5] = sint * sint * sint * sint * sint * cost;
            post[4] = sint * sint * sint * sint * (11.0 * cost * cost - 1.0);
            post[3] = sint * sint * sint * (11.0 * cost * cost * cost - 3.0 * cost);
            post[2] = sint * sint * (33.0 * cost * cost * cost * cost - 18.0 * cost * cost + 1.0);
            post[1] = sint * (33.0 * cost * cost * cost * cost * cost - 30.0 * cost * cost * cost + 5.0 * cost);
            post[0] = 231.0 * cost * cost * cost * cost * cost * cost - 315.0 * cost * cost * cost * cost + 105.0 * cost * cost - 5.0;

            std::complex<double> phase(cosp, sinp);
            std::complex<double> phase_power(1.0, 0.0);
            molecule_Q6[6] += pre[0] * post[0];
            for (int m = 1; m <= 6; ++m) {
                phase_power *= phase;
                molecule_Q6[6 - m] += pre[m] * std::conj(phase_power) * post[m];
                if (m % 2 == 0) {
                    molecule_Q6[6 + m] += pre[m] * phase_power * post[m];
                }
                else {
                    molecule_Q6[6 + m] -= pre[m] * phase_power * post[m];
                }
            }
        }

        for (int m = 0; m < 13; ++m) {
            global_Q6[m] += molecule_Q6[m] / double(Q6_neighbors[mol].size());
        }
    }

    Q6 = 0.0;
    for (int m = 0; m < 13; ++m) {
        global_Q6[m] /= double(Q6_neighbors.size());
        Q6 += std::real(global_Q6[m] * std::conj(global_Q6[m]));
    }
    Q6 = sqrt(Q6);
}
