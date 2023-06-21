#include "particles.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "global.hpp"
#include "read_infile.hpp"
#include "identical_particles.hpp"

// contructor
identical_particles::identical_particles(YAML::Node doc) : particles_type(doc) {
    mass = check_and_assign_value<double>(doc["particles"], "mass");
    beta = check_and_assign_value<double>(doc["particles"], "beta");
    sbeta = sqrt(beta);
    cutoff = check_and_assign_value<double>(doc["particles"], "cutoff");
    eps = check_and_assign_value<double>(doc["particles"], "eps");
    sigma = check_and_assign_value<double>(doc["particles"], "sigma");

    std::string algorithm = check_and_assign_value<std::string>(doc["particles"], "algorithm");
    if (algorithm.compare("all_neighbour") == 0) {
        potential_strategy = std::bind(&identical_particles::potential_all_neighbour, this);
    }
    if (algorithm.compare("binning_serial") == 0) {
        binning_geometry_strategy = std::bind(&identical_particles::cutoff_binning, this);
        binning_geometry();
        binning_strategy = std::bind(&identical_particles::serial_binning, this);
        serial_binning_init();
        potential_strategy = std::bind(&identical_particles::potential_binning, this);
    }
    if (algorithm.compare("parallel_binning") == 0) {
        binning_geometry_strategy = std::bind(&identical_particles::cutoff_binning, this);
        binning_geometry();
        binning_strategy = std::bind(&identical_particles::parallel_binning, this);
        parallel_binning_init();
        potential_strategy = std::bind(&identical_particles::potential_binning, this);
    }
    if (algorithm.compare("quick_sort") == 0) {
        binning_geometry_strategy = std::bind(&identical_particles::cutoff_binning, this);
        binning_geometry();
        binning_strategy = std::bind(&identical_particles::create_quick_sort, this);
        quick_sort_init();
        potential_strategy = std::bind(&identical_particles::potential_binning, this);
    }
    std::cout << "partilces_type:" << std::endl;
    std::cout << "name:" << name << std::endl;
    std::cout << "mass:" << mass << std::endl;
    std::cout << "beta:" << beta << std::endl;

    compute_coeff_momenta();
    compute_coeff_position();
}


void identical_particles::InitX() {
    x = type_x("x", N);

    p = type_p("p", N);
    f = type_f("f", N);

    if (params.StartCondition == "cold") {
        Kokkos::parallel_for("cold initialization", Kokkos::RangePolicy<cold>(0, N), *this);
    }
    else if (params.StartCondition == "hot") {
        Kokkos::parallel_for("hot initialization", Kokkos::RangePolicy<hot>(0, N), *this);
    }
    else {
        Kokkos::abort("StartCondition not implemented");
    }
    // create_mirror() willalways allocate a new view,
    // create_mirror_view() will only create a new view if the original one is not in HostSpace
    h_x = Kokkos::create_mirror(x);
    Kokkos::deep_copy(h_x, x);
    Kokkos::fence();
    printf("particle initialized\n");
};

KOKKOS_FUNCTION
void identical_particles::operator() (cold, const int i) const {
    double N3 = pow(N, 1. / 3.);
    int iz = (int)i / (N3 * N3);
    int iy = (int)(i - iz * N3 * N3) / (N3);
    int ix = (int)(i - iz * N3 * N3 - iy * N3);

    x(i, 0) = params.L[0] * (ix / N3);
    x(i, 1) = params.L[1] * (iy / N3);
    x(i, 2) = params.L[2] * (iz / N3);
};

KOKKOS_FUNCTION
void identical_particles::operator() (hot, const int i) const {
    gen_type rgen = rand_pool.get_state(i);
    x(i, 0) = rgen.drand() * params.L[0];
    x(i, 1) = rgen.drand() * params.L[1];
    x(i, 2) = rgen.drand() * params.L[2];
    rand_pool.free_state(rgen);
};

// since we are using the hostMirror to store the starting point we don't whant to 
// deep_copy it here 
void particles_type::print_xyz(int traj, double K, double V) {
    printf("%d\n", N);
    printf("trajectory= %d,  kinetic_energy= %.12g,  potential= %.12g\n", traj, K, V);
    for (int i = 0; i < N; i++)
        printf("%-10d  %-20.12g %-20.12g %-20.12g\n", i, h_x(i, 0), h_x(i, 1), h_x(i, 2));
}

void particles_type::printx() {
    for (int i = 0; i < N; i++)
        printf("particle(%d)=%-20.12g %-20.12g %-20.12g\n", i, h_x(i, 0), h_x(i, 1), h_x(i, 2));
}
void particles_type::printp() {
    if (!initHostMirror) {
        h_p = Kokkos::create_mirror_view(p);
        initHostMirror = true;
    }
    Kokkos::deep_copy(h_p, p);
    for (int i = 0; i < N; i++)
        printf("momentum(%d)=%-20.12g %-20.12g %-20.12g\n", i, h_p(i, 0), h_p(i, 1), h_p(i, 2));
}


void identical_particles::hb() {
    Kokkos::parallel_for("hb_momenta", Kokkos::RangePolicy<hbTag>(0, N), *this);
}

KOKKOS_FUNCTION
void identical_particles::operator() (hbTag, const int i) const {
    gen_type rgen = rand_pool.get_state(i);
    p(i, 0) = rgen.normal(0, mass / sbeta); // exp(- beta p^2/(2m^2))
    p(i, 1) = rgen.normal(0, mass / sbeta);
    p(i, 2) = rgen.normal(0, mass / sbeta);
    rand_pool.free_state(rgen);
}

double identical_particles::potential_all_neighbour() {
    double result = 0;
    Kokkos::parallel_reduce("identical_particles-LJ-potential-all", Kokkos::RangePolicy<Tag_potential_all>(0, N), *this, result);
    return result;
}


KOKKOS_FUNCTION
void identical_particles::operator() (Tag_potential_all, const int i, double& V) const {

    for (int bx = -1; bx < 2; bx++) {
        for (int by = -1; by < 2; by++) {
            for (int bz = -1; bz < 2; bz++) {
                for (int j = 0; j < N; j++) {   //loop over all distinict pairs i,j

                    double r = 0;

                    double  rij = x(i, 0) - (x(j, 0) + bx * L[0]);
                    r = r + rij * rij;
                    rij = x(i, 1) - (x(j, 1) + by * L[1]);
                    r = r + rij * rij;
                    rij = x(i, 2) - (x(j, 2) + bz * L[2]);
                    r = r + rij * rij;
                    //printf("%g\n",rij);

                    r = sqrt(r);
                    if (r < cutoff && !(i == j && bx == 0 && by == 0 && bz == 0)) {
                        // printf("r(%d,%d)=%g\n", i, j, r);
                        V += 4 * eps * (pow(sigma / r, 12) - pow(sigma / r, 6));
                    }
                }
            }
        }
    }

};


double identical_particles::potential_binning() {
    double result = 0;
    create_binning();
    typedef Kokkos::TeamPolicy<Tag_potential_binning>  team_policy;
    Kokkos::parallel_reduce("identical_particles-LJ-potential-binning", team_policy(bintot, Kokkos::AUTO), *this, result);
    return result;
}

KOKKOS_FUNCTION
void identical_particles::operator() (Tag_potential_binning, const member_type& teamMember, double& V) const {
    const int ib = teamMember.league_rank();// bin id
    double tempN = 0;
    // printf("bin =%d binncount=%d\n", ib, bincount(ib));
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, bincount(ib)), [=](const int ip, double& innerUpdateN) {
        // printf("calling purmute_vector %d\n", ip + binoffsets(ib));
        int i = permute_vector(ip + binoffsets(ib));
        // printf("purmute_vector called\n");
        int ibx, iby, ibz;
        lextoc(ib, ibx, iby, ibz);

        for (int bx = -1; bx < 2; bx++) {
            for (int by = -1; by < 2; by++) {
                for (int bz = -1; bz < 2; bz++) {
                    int jbx = ibx + bx;
                    int jby = iby + by;
                    int jbz = ibz + bz;
                    double wrap_x = 0, wrap_y = 0, wrap_z = 0;
                    if (jbx < 0 || jbx >= nbin[0]) {
                        jbx = (ibx + nbin[0] + bx) % nbin[0];
                        wrap_x = bx * L[0];
                    }
                    if (jby < 0 || jby >= nbin[1]) {
                        jby = (iby + nbin[1] + by) % nbin[1];
                        wrap_y = by * L[1];
                    }
                    if (jbz < 0 || jbz >= nbin[2]) {
                        jbz = (ibz + nbin[2] + bz) % nbin[2];
                        wrap_z = bz * L[2];
                    }
                    int jb = ctolex(jbx, jby, jbz);
                    double tempM = 0;
                    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(teamMember, bincount(jb)), [=](const int jp, double& innerUpdateM) {
                        int j = permute_vector(jp + binoffsets(jb));

                        if (!(i == j && bx == 0 && by == 0 && bz == 0)) {
                            double  r = sqrt((x(i, 0) - x(j, 0) - wrap_x) * (x(i, 0) - x(j, 0) - wrap_x) +
                                (x(i, 1) - x(j, 1) - wrap_y) * (x(i, 1) - x(j, 1) - wrap_y) +
                                (x(i, 2) - x(j, 2) - wrap_z) * (x(i, 2) - x(j, 2) - wrap_z));
                            if (r < cutoff) {
                                // printf("r(%d,%d)=%g\n", i, j, r);
                                innerUpdateM += 4 * eps * (pow(sigma / r, 12) - pow(sigma / r, 6));
                            }
                        }
                        }, tempM);
                    // Kokkos::single(Kokkos::PerThread(teamMember), [=]() {
                    innerUpdateN += tempM;
                    // });
                }
            }
        }
        }, tempN);
    Kokkos::single(Kokkos::PerTeam(teamMember), [&]() {
        V += tempN;
        });
};




double identical_particles::compute_kinetic_E() {
    double K = 0;
    Kokkos::parallel_reduce("identical-particles-LJ-kinetic-E", Kokkos::RangePolicy<kinetic>(0, N), *this, K);
    return K;
}

KOKKOS_FUNCTION
void identical_particles::operator() (kinetic, const int i, double& K) const {
    K += (beta / (2 * mass * mass)) * (p(i, 0) * p(i, 0) + p(i, 1) * p(i, 1) + p(i, 2) * p(i, 2));
};




void identical_particles::compute_force() {
    Kokkos::parallel_for("identical_particles-LJ-force", Kokkos::RangePolicy<force>(0, N), *this);
}

KOKKOS_FUNCTION
void identical_particles::operator() (force, const int i) const {

    f(i, 0) = 0;
    f(i, 1) = 0;
    f(i, 2) = 0;
    for (int bx = -1; bx < 2; bx++) {
        for (int by = -1; by < 2; by++) {
            for (int bz = -1; bz < 2; bz++) {
                for (int j = 0; j < N; j++) {   //loop over all distinict pairs i,j

                    double r = 0;

                    double  rij = x(i, 0) - (x(j, 0) + bx * L[0]);
                    r = r + rij * rij;
                    rij = x(i, 1) - (x(j, 1) + by * L[1]);
                    r = r + rij * rij;
                    rij = x(i, 2) - (x(j, 2) + bz * L[2]);
                    r = r + rij * rij;
                    //printf("%g\n",rij);

                    r = sqrt(r);
                    if (r < cutoff && !(i == j && bx == 0 && by == 0 && bz == 0)) {
                        f(i, 0) += 4 * eps * (pow(sigma / r, 10) - pow(sigma / r, 4)) * x(i, 0);
                        f(i, 1) += 4 * eps * (pow(sigma / r, 10) - pow(sigma / r, 4)) * x(i, 1);
                        f(i, 2) += 4 * eps * (pow(sigma / r, 10) - pow(sigma / r, 4)) * x(i, 2);
                    }
                }
            }
        }
    }


}


void identical_particles::compute_coeff_momenta() {
    coeff_p = 1;
}

void identical_particles::compute_coeff_position() {
    coeff_x = -beta / (mass * mass);
}


class functor_update_pos {
public:
    const double dt;
    const double c;
    type_x x;
    type_const_p p;
    const double L[dim_space];
    functor_update_pos(double dt_, double c, type_x& x_, type_p& p_, const double L_[]) : dt(dt_), c(c), x(x_), p(p_),
        L{ L_[0], L_[1], L_[2] } {
    };

    KOKKOS_FUNCTION
        void operator() (const int i) const {
        for (int dir = 0; dir < 3;dir++) {
            x(i, dir) += dt * c * p(i, 0);
            // apply boundary condition
            x(i, dir) -= L[dir] * floor(x(i, dir) / L[dir]);
        }
    };
};
void  identical_particles::update_positions(const double dt_) {
    Kokkos::parallel_for("update_position", Kokkos::RangePolicy(0, N), functor_update_pos(dt_, coeff_x, x, p, L));
}


class functor_update_momenta {
public:
    const double dt;
    type_p p;
    type_const_f f;
    functor_update_momenta(double dt_, type_p& p_, type_f& f_) : dt(dt_), p(p_), f(f_) {};

    KOKKOS_FUNCTION
        void operator() (const int i) const {
        p(i, 0) -= dt * f(i, 0);
        p(i, 1) -= dt * f(i, 1);
        p(i, 2) -= dt * f(i, 2);
    };
};
void identical_particles::update_momenta(const double dt_) {
    compute_force();
    Kokkos::parallel_for("update_momenta", Kokkos::RangePolicy(0, N), functor_update_momenta(dt_, p, f));
}