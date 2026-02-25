#define CONTROL

#include <Kokkos_Core.hpp>
#include <memory>
#include <iostream>
#include <fstream>

#include "git_version.hpp"
#include "HMC.hpp"
#include "global.hpp"
#include "Input_reader.hpp"
#include "Parameters.hpp"
#include "Calc_Manager.hpp"
#include "particles.hpp"

void run_pot_test(HMC_class& HMC) {

    auto& calc_manager = HMC.calc_manager;
    auto& integrator = HMC.integrator;
    auto& particles = integrator->particles;

    double tokcal = 1.0/kcaltointernal;

    double Vi = calc_manager->compute_potential();
    printf("INITIAL V: %f \n", Vi*tokcal);

    // initialize view for numerical difference of forces to 0
    Kokkos::View<double*[3]> num_f =  type_f("f", particles->N);
    Kokkos::deep_copy(num_f,0.0);

    constexpr double diff = 1e-5; // displacement for particle diffs

    calc_manager->compute_force(); // analytic forces are now in particles->f
    Kokkos::fence();

    auto& f = particles->f;

    double V_1;
    double V_2;
    
    for (size_t i = 0; i < num_f.extent(0); i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            particles->h_x(i,j) += diff;
            Kokkos::deep_copy(particles->x,particles->h_x);
            V_1 = calc_manager->compute_potential();
            Kokkos::fence();
            particles->h_x(i,j) -= 2 * diff;
            Kokkos::deep_copy(particles->x,particles->h_x);
            V_2 = calc_manager->compute_potential();
            Kokkos::fence();
            //printf("%d %d 1:%f 2:%f\n",i,j,V_1,V_2);
            particles->x(i,j) += diff; // restore original position
            num_f(i,j) = (V_1 - V_2) / (2.0 * diff);
            particles->h_x(i,j) += diff;
        }
        printf("Atom %lu: %f|%f \t %f|%f \t %f|%f\n", i,
            f(i,0), num_f(i,0),
            f(i,1), num_f(i,1),
            f(i,2), num_f(i,2));
        printf("Atom %lu DIFF: %f \t %f \t %f\n", i,
            (f(i,0)- num_f(i,0))*evtointernal/kcaltointernal,
            (f(i,1)- num_f(i,1))*evtointernal/kcaltointernal,
            (f(i,2)- num_f(i,2))*evtointernal/kcaltointernal);
    }

    printf("index: ana|num\n");
}

void run_md_toteng_test(HMC_class& HMC) {

    auto& calc_manager = HMC.calc_manager;
    auto& integrator = HMC.integrator;
    auto& particles = integrator->particles;

    double dt = 1.0;
    size_t steps = 1000;

    double tokcal = 1.0/kcaltointernal;

    particles->hb();

    double initial_H = particles->compute_kinetic_E() + calc_manager->compute_potential();
    double H,T,V;
    printf("Initial H: %f\n",initial_H*tokcal);
    printf("Initial T: %f\n",particles->compute_kinetic_E()*tokcal);

    for (size_t i = 0; i < steps; i++) {
        calc_manager->compute_force();
        Kokkos::fence();
        particles->update_momenta(dt / 2.0);
        Kokkos::deep_copy(particles->h_p, particles->p);
        particles->update_positions(dt);
        Kokkos::deep_copy(particles->h_x, particles->x);
        if (particles->neighbor_list_used) {
            particles->neighbor_list->build_verlet_list(*particles);
        }
        calc_manager->compute_force();
        Kokkos::fence();
        particles->update_momenta(dt / 2.0);
        T = particles->compute_kinetic_E();
        V = calc_manager->compute_potential();
        H = T + V;
        printf("Step: %d H: %f \t T: %f\t V: %f\n",i,H*tokcal, T*tokcal, V*tokcal);
    }
    H = particles->compute_kinetic_E() + calc_manager->compute_potential();
    //printf("H DIFF: %f\n",(H - initial_H)*tokcal);
}

void run_minimization_test(HMC_class& HMC) {
    auto& calc_manager = HMC.calc_manager;
    auto& integrator = HMC.integrator;
    auto& particles = integrator->particles;

    calc_manager->minimize_energy(HMC.doc);
}

int main(int argc, char** argv) {

    printf("chemHMC git commit %s\n", kGitHash);

    // starting kokkos
    Kokkos::initialize(argc, argv); {
        Kokkos::Timer timer;

        HMC_class HMC;
        HMC.init(argc, argv);

        //run_pot_test(HMC);
        //run_minimization_test(HMC);
        run_md_toteng_test(HMC);
        
        printf("total test time = %f s\n", timer.seconds());
    }
    Kokkos::finalize();
}

/*void run() {

    Kokkos::Timer timer;
    double tokcal = 1.0/kcaltointernal;

    double Vi = calc_manager->compute_potential();
    printf("INITIAL V: %f \n", Vi*tokcal);

    double beta = integrator->particles->get_beta();
    calc_manager->compute_force();

    Kokkos::fence();
    // copy the configuration before the MD
    Kokkos::deep_copy(integrator->particles->h_x, integrator->particles->x);// h_x=x;
    for (int i = 1; i <= params->Ntrajectories; i++) {
        Kokkos::Timer timer_traj;
        // hb momenta
        integrator->particles->hb();
        double Ki = integrator->particles->compute_kinetic_E();
        // molecular dynamics
        integrator->integrate();

        // accept/reject
        double Vf = calc_manager->compute_potential();
        double Kf = integrator->particles->compute_kinetic_E();

        double dh = beta * (Kf + Vf - Ki - Vi);
        double exp_mdh = exp(-dh);

        if ((i % params->print_info_every == 0)) {
            printf("step %d: K = %.12g  V = %.12g exp_mdh = %.12g\n", i, Kf*tokcal, Vf*tokcal, exp_mdh);
        }
        Kokkos::fence();

        if (i < params->thermalization_steps) {
            Vi = Vf;
            Ki = Kf;
            Kokkos::deep_copy(integrator->particles->h_x, integrator->particles->x);// h_x=x;
        }
        else {
            double r = gen_random();// random number from 0 to 1
            if (r < exp_mdh) {
                acceptance++;
                Vi = Vf;
                Ki = Kf;
                Kokkos::deep_copy(integrator->particles->h_x, integrator->particles->x);// h_x=x;
            }
            else {
                Kokkos::deep_copy(integrator->particles->x, integrator->particles->h_x);
            }
            // save
            if ((i % params->save_every == 0)) {
                integrator->particles->print_xyz(*params, i, Ki, Vi);
            }
        }
    }
    printf("Acceptance: %g\n", acceptance / ((double)(params->Ntrajectories - params->thermalization_steps)));
    //printf("final step size: %f\n", integrator->dt);
    printf("time for HMC: %g  s\n", timer.seconds());
    calc_manager->print_timings();
    if (integrator->particles->neighbor_list_used) {
        printf("time for Neighbor list builds: %g s\n",particles->neighbor_list->time_list_build);
    }
}*/