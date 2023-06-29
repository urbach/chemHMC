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
    name_xyz = check_and_assign_value<std::string>(doc["particles"], "name_xyz");

    std::string algorithm = check_and_assign_value<std::string>(doc["particles"], "algorithm");
    if (algorithm.compare("all_neighbour") == 0) {
        potential_strategy = std::bind(&identical_particles::potential_all_neighbour, this);
        force_strategy = std::bind(&identical_particles::compute_force_all, this);
    }
    if (algorithm.compare("binning_serial") == 0) {
        binning_geometry_strategy = std::bind(&identical_particles::cutoff_binning, this);
        binning_geometry();
        binning_strategy = std::bind(&identical_particles::serial_binning, this);
        serial_binning_init();
        potential_strategy = std::bind(&identical_particles::potential_binning, this);
        force_strategy = std::bind(&identical_particles::compute_force_binning, this);
    }
    if (algorithm.compare("parallel_binning") == 0) {
        binning_geometry_strategy = std::bind(&identical_particles::cutoff_binning, this);
        binning_geometry();
        binning_strategy = std::bind(&identical_particles::parallel_binning, this);
        parallel_binning_init();
        potential_strategy = std::bind(&identical_particles::potential_binning, this);
        force_strategy = std::bind(&identical_particles::compute_force_binning, this);
    }
    if (algorithm.compare("quick_sort") == 0) {
        binning_geometry_strategy = std::bind(&identical_particles::cutoff_binning, this);
        binning_geometry();
        binning_strategy = std::bind(&identical_particles::create_quick_sort, this);
        quick_sort_init();
        potential_strategy = std::bind(&identical_particles::potential_binning, this);
        force_strategy = std::bind(&identical_particles::compute_force_binning, this);
    }
    std::cout << "partilces_type:" << std::endl;
    std::cout << "name:" << name << std::endl;
    std::cout << "mass:" << mass << std::endl;
    std::cout << "beta:" << beta << std::endl;

    compute_coeff_momenta();
    compute_coeff_position();
}

void identical_particles::read_xyz() {
    FILE* file = NULL;
    file = fopen(params.start_configuration_file.c_str(), "r");
    if (file == NULL) {
        printf("error in opening file %s\n", params.start_configuration_file.c_str());
        Kokkos::abort("abort");
    }
    int lines = 0;
    char c;

    /* count the newline characters */
    while ((c = fgetc(file)) != EOF) {
        if (c == '\n')
            lines++;
    }
    if (lines % (N + 2) != 0) {
        printf("error: input file %s contains %d lines\n", params.start_configuration_file.c_str(), lines);
        printf("       the number of lines mus be a multiple of N+2=%d\n", N + 2);
        Kokkos::abort("abort");
    }
    int confs = lines / (N + 2);
    printf("confs in input configuration file %d\n", confs);
    // go to last configuration and read it 
    rewind(file);
    int count = 0;
    char id[1000];

    while ((c = fgetc(file)) != EOF) {

        if (c == '\n') {
            count++;
            if (count == (confs - 1) * (N + 2) + 1) {
                for (int i = 0;i < 11;i++) c = fgetc(file);
                fscanf(file, " %d",  &params.istart);
                printf("%d %d\n",  params.istart, count);
                // count++;
            }
            if (count == (confs - 1) * (N + 2) + 2) {// if starting of the last conf, count missmatched by fscanf
                break;
            }
        }
    }
    printf("reading last configuration from input file %s\n", params.start_configuration_file.c_str());
    count = 0;
    for (int i = 0; i < N;i++) {
        count += fscanf(file, "%s   %lf   %lf  %lf\n", id, &h_x(i, 0), &h_x(i, 1), &h_x(i, 2));
        // printf("%s   %lf   %lf  %lf\n", id, h_x(i, 0), h_x(i, 1), h_x(i, 2));
    }
    if (name_xyz.compare(id) != 0) {
        printf("name in the xyz file: %s  do not mach the name in the input file: %s\n", id, name_xyz.c_str());
        Kokkos::abort("abort");
    }
    // printf("%d  %d\n", count, N);
    if (count != N * 4) { Kokkos::abort("error in reading the file"); }
    fclose(file);
    Kokkos::deep_copy(x, h_x);
    printx();
}

void identical_particles::InitX() {
    x = type_x("x", N);
    // create_mirror() will always allocate a new view,
    // create_mirror_view() will only create a new view if the original one is not in HostSpace
    h_x = Kokkos::create_mirror(x);
    p = type_p("p", N);
    f = type_f("f", N);

    if (params.StartCondition == "cold") {
        Kokkos::parallel_for("cold initialization", Kokkos::RangePolicy<cold>(0, N), *this);
    }
    else if (params.StartCondition == "hot") {
        Kokkos::parallel_for("hot initialization", Kokkos::RangePolicy<hot>(0, N), *this);
    }
    else if (params.StartCondition == "read") {
        read_xyz();
    }
    else {
        Kokkos::abort("StartCondition not implemented");
    }

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
void identical_particles::print_xyz(int traj, double K, double V) {
    fprintf(params.fileout, "     %d\n", N);
    fprintf(params.fileout, "trajectory= %d  kinetic_energy= %.12g  potential= %.12g\n", traj, K, V);
    for (int i = 0; i < N; i++)
        fprintf(params.fileout, "%s  %-20.12g %-20.12g %-20.12g\n", name_xyz.c_str(), h_x(i, 0), h_x(i, 1), h_x(i, 2));
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
    return 4 * eps * result;
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

                    r = sqrt(r);
                    if (r < cutoff && !(i == j && bx == 0 && by == 0 && bz == 0)) {
                        V += (pow(sigma / r, 12) - pow(sigma / r, 6));
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
    return 4 * eps * result;
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
                                // we multiply by 4*eps at the end, outside this operator
                                double sr = sigma / r;
                                double sr6 = sr * sr * sr * sr * sr * sr;
                                innerUpdateM += sr6 * (sr6 - 1.0);
                                // innerUpdateM +=  (pow(sigma / r, 12) - pow(sigma / r, 6));
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
    K += (p(i, 0) * p(i, 0) + p(i, 1) * p(i, 1) + p(i, 2) * p(i, 2)) / (2 * mass * mass);
};




void identical_particles::compute_force_all() {
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
                        f(i, 0) += (pow(sigma / r, 10) - pow(sigma / r, 4)) * x(i, 0);
                        f(i, 1) += (pow(sigma / r, 10) - pow(sigma / r, 4)) * x(i, 1);
                        f(i, 2) += (pow(sigma / r, 10) - pow(sigma / r, 4)) * x(i, 2);
                    }
                }
            }
        }
    }
    f(i, 0) *= 4 * eps;
    f(i, 1) *= 4 * eps;
    f(i, 2) *= 4 * eps;
}





void identical_particles::compute_force_binning() {
    create_binning();
    typedef Kokkos::TeamPolicy<Tag_force_binning>  team_policy;
    Kokkos::parallel_for("identical_particles-LJ-force-binning", team_policy(bintot, Kokkos::AUTO), *this);
}

// we need a ruduction of 3 double array
template< class ScalarType, int N >
struct array_type {
    ScalarType the_array[N];

    KOKKOS_INLINE_FUNCTION   // Default constructor - Initialize to 0's
        array_type() {
        for (int i = 0; i < N; i++) { the_array[i] = 0; }
    }
    KOKKOS_INLINE_FUNCTION   // Copy Constructor
        array_type(const array_type& rhs) {
        for (int i = 0; i < N; i++) {
            the_array[i] = rhs.the_array[i];
        }
    }
    KOKKOS_INLINE_FUNCTION   // add operator
        array_type& operator += (const array_type& src) {
        for (int i = 0; i < N; i++) {
            the_array[i] += src.the_array[i];
        }
        return *this;
    }
};
typedef array_type<double, dim_space> space_vector;  // used to simplify code below
namespace Kokkos { //reduction identity must be defined in Kokkos namespace
    template<>
    struct reduction_identity< space_vector > {
        KOKKOS_FORCEINLINE_FUNCTION static space_vector sum() {
            return space_vector();
        }
    };
}
KOKKOS_FUNCTION
void identical_particles::operator() (Tag_force_binning, const member_type& teamMember) const {
    const int ib = teamMember.league_rank();// bin id
    // printf("bin =%d binncount=%d\n", ib, bincount(ib));
    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, bincount(ib)), [=](const int ip) {
        // printf("calling purmute_vector %d\n", ip + binoffsets(ib));
        int i = permute_vector(ip + binoffsets(ib));
        f(i, 0) = 0;
        f(i, 1) = 0;
        f(i, 2) = 0;
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
                    space_vector  fv;
                    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(teamMember, bincount(jb)), [=](const int jp, space_vector& innerfv) {
                        int j = permute_vector(jp + binoffsets(jb));

                        if (!(i == j && bx == 0 && by == 0 && bz == 0)) {
                            double  r = sqrt((x(i, 0) - x(j, 0) - wrap_x) * (x(i, 0) - x(j, 0) - wrap_x) +
                                (x(i, 1) - x(j, 1) - wrap_y) * (x(i, 1) - x(j, 1) - wrap_y) +
                                (x(i, 2) - x(j, 2) - wrap_z) * (x(i, 2) - x(j, 2) - wrap_z));
                            if (r < cutoff) {
                                double sr = sigma / r;
                                double sr4 = sr * sr * sr * sr;
                                sr = sr4 * (sr4 * sr * sr - 1.0);
                                innerfv.the_array[0] += sr * x(i, 0);
                                innerfv.the_array[1] += sr * x(i, 1);
                                innerfv.the_array[2] += sr * x(i, 2);
                                // innerfv.the_array[0] += (pow(sigma / r, 10) - pow(sigma / r, 4)) * x(i, 0);
                                // innerfv.the_array[1] += (pow(sigma / r, 10) - pow(sigma / r, 4)) * x(i, 1);
                                // innerfv.the_array[2] += (pow(sigma / r, 10) - pow(sigma / r, 4)) * x(i, 2);
                            }
                        }
                        }, fv);
                    // Kokkos::single(Kokkos::PerThread(teamMember), [=]() {
                    f(i, 0) += fv.the_array[0];
                    f(i, 1) += fv.the_array[1];
                    f(i, 2) += fv.the_array[2];
                    // });
                }
            }
        }
        f(i, 0) *= 4.0 * eps;
        f(i, 1) *= 4.0 * eps;
        f(i, 2) *= 4.0 * eps;
        });

    // Kokkos::single(Kokkos::PerTeam(teamMember), [&]() {
    //     V += tempN;
    //     });
};


void identical_particles::compute_coeff_momenta() {
    coeff_p = beta;
}

void identical_particles::compute_coeff_position() {
    coeff_x = beta / (mass * mass);
}


class functor_update_pos {
public:
    const double dt;
    const double c;
    type_x x;
    type_const_p p;
    const double L[dim_space];
    functor_update_pos(double dt_, double c_, type_x& x_, type_p& p_, const double L_[]) : dt(dt_), c(c_), x(x_), p(p_),
        L{ L_[0], L_[1], L_[2] } {
    };

    KOKKOS_FUNCTION
        void operator() (const int i) const {
        for (int dir = 0; dir < 3;dir++) {
            x(i, dir) += dt * c * p(i, 0);
            // apply  periodic boundary condition
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
    const double c;
    type_p p;
    type_const_f f;
    functor_update_momenta(double dt_, double c_, type_p& p_, type_f& f_) : dt(dt_), c(c_), p(p_), f(f_) {};

    KOKKOS_FUNCTION
        void operator() (const int i) const {
        p(i, 0) -= dt * c * f(i, 0);
        p(i, 1) -= dt * c * f(i, 1);
        p(i, 2) -= dt * c * f(i, 2);
    };
};
void identical_particles::update_momenta(const double dt_) {
    compute_force();
    Kokkos::parallel_for("update_momenta", Kokkos::RangePolicy(0, N), functor_update_momenta(dt_, coeff_p, p, f));
}