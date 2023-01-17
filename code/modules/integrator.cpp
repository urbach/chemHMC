#include "integrator.hpp"
#include "read_infile.hpp"


integrator_type::integrator_type(YAML::Node doc) {

    if (doc["particles"]) {
        std::string name = check_and_assign_value<std::string>(doc["particles"], "name");
        if (name == "identical_particles")
            particles = new identical_particles(doc);
        else {
            printf("no valid name for particles: ");
            std::cout << doc["particles"].as<std::string>() << std::endl;
            exit(1);
        }
        particles->InitX();
    }
    else {
        Kokkos::abort("no particles in input file");
    }

    steps = check_and_assign_value<int>(doc["integrator"], "steps");
    dt = check_and_assign_value<double>(doc["integrator"], "dt");
    std::cout << "steps: " << steps << std::endl;
    std::cout << "dt: " << dt << std::endl;
    N = particles->N;

}

LEAP::LEAP(YAML::Node doc): integrator_type(doc) {

}



class functor_update_momenta {
public:
    const double dt;
    type_p p;
    type_const_f f;
    functor_update_momenta(double dt_, type_p& p_, type_f& f_): dt(dt_), p(p_), f(f_) {};

    KOKKOS_FUNCTION
        void operator() (const int i) const {
        p(i, 0) -= dt * f(i, 0);
        p(i, 1) -= dt * f(i, 1);
        p(i, 2) -= dt * f(i, 2);
    };
};
void LEAP::update_momenta(const double dt_) {
    particles->compute_force();
    Kokkos::parallel_for("update_momenta", Kokkos::RangePolicy(0, N), functor_update_momenta(dt_, particles->p, particles->f));
}



class functor_update_pos {
public:
    const double dt;
    const double c;
    type_x x;
    type_const_p p;
    functor_update_pos(double dt_, double c, type_x& x_, type_p& p_): dt(dt_), c(c), x(x_), p(p_) {};

    KOKKOS_FUNCTION
        void operator() (const int i) const {
        x(i, 0) += dt * c * p(i, 0);
        x(i, 1) += dt * c * p(i, 1);
        x(i, 2) += dt * c * p(i, 2);
        printf("c=%g   dt=%g   p=%g  %g  %g\n", c, dt, p(i, 0), p(i, 1), p(i, 2));
    };
};
void  LEAP::update_positions(const double dt_) {
    Kokkos::parallel_for("update_momenta", Kokkos::RangePolicy(0, N), functor_update_pos(dt_, particles->coeff_x, particles->x, particles->p));
}


void LEAP::integrate() {

    // initial half-step for the  momenta
    update_momenta(dt / 2.);
    // first full step for the position
    update_positions(dt);
    // nsteps-1 full steps
    for (size_t i = 0; i < steps - 1; i++) {
        update_momenta(dt);
        update_positions(dt);
    }
    // final half-step for the momenta
    update_momenta(dt / 2.);
}


//////////////////////////////////////////////////////////////////////////////
// OMF2
//////////////////////////////////////////////////////////////////////////////
OMF2::OMF2(YAML::Node doc): integrator_type(doc), lambda(0.1938), oneminus2lambda(1. - 2. * lambda) {
}

void OMF2::update_momenta(const double dt_) {
    particles->compute_force();
    Kokkos::parallel_for("update_momenta", Kokkos::RangePolicy(0, N), functor_update_momenta(dt_, particles->p, particles->f));
}
void  OMF2::update_positions(const double dt_) {
    Kokkos::parallel_for("update_momenta", Kokkos::RangePolicy(0, N), functor_update_pos(dt_, particles->coeff_x, particles->x, particles->p));
}


void OMF2::integrate() {

    // initial half-step for the  momenta
    update_momenta(lambda * dt);

    // nsteps-1 full steps
    for (size_t i = 0; i < steps - 1; i++) {
        update_positions(dt / 2.);
        update_momenta(oneminus2lambda * dt);
        update_positions(dt / 2.);
        update_momenta(2. * lambda * dt);
    }
    // final step
    update_positions(dt / 2.);
    update_momenta(oneminus2lambda * dt);
    update_positions(dt / 2.);
    update_momenta( lambda * dt);
}