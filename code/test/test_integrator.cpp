#define CONTROL

#include <Kokkos_Core.hpp>
#include "global.hpp"
#include "git_version.hpp"
#include "HMC.hpp"
#include "identical_particles.hpp"

void add_error(std::vector<std::string>& errors, std::string s) {
    errors.emplace_back(s);
    printf("%s\n", s.c_str());
}


void check_integrator_res(integrator_type* int_1, integrator_type* int_2, std::string comparison, std::vector<std::string>& errors) {
    printf("########################################################################################\n");
    printf("comparing position after integrator %s\n", comparison.c_str());
    double sum = 0;
    int N = int_1->particles->N;
    type_x f2 = int_1->particles->x;
    type_x f3 = int_2->particles->x;
    Kokkos::parallel_reduce("check-integrator", N, KOKKOS_LAMBDA(const int i, double& update) {
        double diff;
        for (int dir = 0;dir < dim_space;dir++) {
            diff = 0;
            if (f2(i, dir) * f2(i, dir) > 1e-6)
                diff = Kokkos::fabs(f2(i, dir) - f3(i, dir)) / f2(i, dir);
            else
                diff = Kokkos::fabs(f2(i, dir) - f3(i, dir));
            if (diff > 1e-7) {
                // printf("error: position difference at %d  f2= %.12g  f3=  %.12g  diff=%.12g  ratio=%.12g\n", i, f2(i, dir), f3(i, dir),
                // f2(i, dir) - f3(i, dir), f2(i, dir) / f3(i, dir));
                update+=diff;
            }
        }

    }, sum);
    Kokkos::fence();
    sum/=N;
    if (sum > 0) {
           add_error(errors, "comparing integrator" + comparison);
           printf("|x1-x2|/N=%g\n",sum);
    }
    else printf("Test passed:  integrators match\n");

}

int main(int argc, char** argv) {

    printf("chemHMC git commit %s\n", kGitHash);

    // starting kokkos
    Kokkos::initialize(argc, argv); {
        std::vector<std::string> errors(0);
        YAML::Node doc = read_params(argc, argv);
        params_class params(doc, false);
        
        integrator_type* int_OMF4 = new OMF4(doc, params);
        integrator_type* int_OMF2 = new OMF2(doc, params);
        integrator_type* int_LEAP = new LEAP(doc, params);
        /////////////////////////////////////////////////////////////////////////////////////////
        printf("########################################################################################\n");
        std::vector<double> times(3);
        double Ki,Vi,Kf,Vf;
        int_OMF4->particles->hb();
        Ki = int_OMF4->particles->compute_kinetic_E();
        Vi = int_OMF4->particles->compute_potential();
        printf("Hi=%g   ",Ki+Vi);
        Kokkos::Timer timer4;
        int_OMF4->integrate();
        times[0]=timer4.seconds();
        Kf = int_OMF4->particles->compute_kinetic_E();
        Vf = int_OMF4->particles->compute_potential();
        printf("Hf_OMF4=%g     diff=%g\n",Kf+Vf,Kf+Vf-(Ki+Vi));
        
        int_OMF2->particles->hb();
        Ki = int_OMF2->particles->compute_kinetic_E();
        Vi = int_OMF2->particles->compute_potential();
        printf("Hi=%g   ",Ki+Vi);
        Kokkos::Timer timer2;
        int_OMF2->integrate();
        times[1]=timer2.seconds();
        Kf = int_OMF2->particles->compute_kinetic_E();
        Vf = int_OMF2->particles->compute_potential();
        printf("Hf_OMF2=%g     diff=%g\n",Kf+Vf,Kf+Vf-(Ki+Vi));
        

        int_LEAP->particles->hb();
        Ki = int_LEAP->particles->compute_kinetic_E();
        Vi = int_LEAP->particles->compute_potential();
        printf("Hi=%g   ",Ki+Vi);
        Kokkos::Timer timerLEAP;
        int_LEAP->integrate();
        times[2]=timerLEAP.seconds();
        Kf = int_LEAP->particles->compute_kinetic_E();
        Vf = int_LEAP->particles->compute_potential();
        printf("Hf_LEAP=%g     diff=%g\n",Kf+Vf,Kf+Vf-(Ki+Vi));

        printf("########################################################################################\n");
        printf("time OMF4 integration = %f s\n", times[0]);
        printf("time OMF2 integration = %f s\n", times[1]);
        printf("time LEAP integration = %f s\n", times[2]);
        /////////////////////////////////////////////////////////////////////////////////////////
        check_integrator_res(int_OMF4, int_OMF2, " OMF4  agains OMF2", errors);
        check_integrator_res(int_OMF2, int_LEAP, " OMF4  agains LEAP", errors);

        //////////////////////////////////////////////////////////////////////////////////////////
        printf("error recap:\n");
        if (errors.size() > 0) {
            for (auto e : errors)
                printf("%s\n", e.c_str());
            Kokkos::abort("abort");
        }
        else {
            printf("none\n");
        }
    }
    Kokkos::finalize();
    printf("all tests passed\n");
}
