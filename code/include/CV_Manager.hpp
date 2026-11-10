#ifndef CV_MANAGER_H
#define CV_MANAGER_H

#include "Calc.hpp"
#include <vector>

class CV_Manager : public Calc {
public:
    double density = 0.0;
    double Q6 = 0.0;

    CV_Manager() = default;
    ~CV_Manager() override = default;

    void init(const particles_instance& particles) override;
    double potential(const particles_instance& particles) override;
    void force(const particles_instance& particles, type_f& f) override;
    void compute_CVs(const particles_instance& particles);
private:
    struct Q6Neighbor {
        std::size_t molecule;
        double displacement[3];
        double distance;
    };

    void compute_molecular_centers(const particles_instance& particles);
    void build_Q6_neighbor_list(const particles_instance& particles);
    void compute_Q6();

    std::vector<std::vector<std::size_t>> molecule_atoms;
    std::vector<double> molecular_centers;
    std::vector<std::vector<Q6Neighbor>> Q6_neighbors;
    Kokkos::View<double**, Kokkos::HostSpace> x_cv;
};

#endif
