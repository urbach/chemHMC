#include "global.hpp"
#include "particles.hpp"

class Coulomb : public Calc {
public:
    // Ewald sum
    double ewald_alpha; // width of the gaussians
    double sqrt_ewald_alpha;
    double r_c; // real-space cutoff for the ewald sum
    double r_c2;
    double ewald_accuracy; // rms accuracy of the ewald sum
    int k_max;
    double V_self = 0.0; // self interaction energy in the ewald sum is precomputed and stored.
    Kokkos::View<double*> charge;
    Kokkos::View<double*>::HostMirror h_charge;

    void init(particles_instance& particles) override;
    double potential(particles_instance& particles) override;
    void force(particles_instance& particles, type_f& f) override;
}