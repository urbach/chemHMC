#define CONTROL

#include <Kokkos_Core.hpp>
#include "global.hpp"
#include "git_version.hpp"
#include "HMC.hpp"
#include "particles.hpp"

void add_error(std::vector<std::string>& errors, std::string s) {
    errors.emplace_back(s);
    printf("%s\n", s.c_str());
}

void check_force(particles_type* particles2, particles_type* particles3, std::string comparison, std::vector<std::string>& errors) {
    printf("########################################################################################\n");
    printf("comparing force %s\n", comparison.c_str());
    int sum = 0;
    int N = particles2->N;
    type_f f2 = particles2->f;
    type_f f3 = particles3->f;
    Kokkos::parallel_reduce("check-force", N, KOKKOS_LAMBDA(const int i, int& update) {
        double diff;
        for (int dir = 0;dir < dim_space;dir++) {
            diff = 0;
            if (f2(i, dir) * f2(i, dir) > 1e-6)
                diff = Kokkos::fabs(f2(i, dir) - f3(i, dir)) / f2(i, dir);
            else
                diff = Kokkos::fabs(f2(i, dir) - f3(i, dir));
            if (diff > 1e-7) {
                printf("error: force difference at %d  f2= %.12g  f3=  %.12g  diff=%.12g  ratio=%.12g\n", i, f2(i, dir), f3(i, dir),
                    f2(i, dir) - f3(i, dir), f2(i, dir) / f3(i, dir));
                update++;
            }
        }

    }, sum);
    Kokkos::fence();

    if (sum > 0)    add_error(errors, "comparing force" + comparison);
    else printf("Test passed:  force match\n");

}

void check_force_with_num_der(particles_type* particles, std::vector<std::string>& errors) {
    printf("###################################################################################################\n");
    printf("compare force and derivative of the potential  %s \n", particles->algorithm.c_str());
    double h = 1e-6;
    type_x tmpx = particles->x;
    int count = 0;
    type_f::HostMirror force_val = Kokkos::create_mirror(particles->f);// force is already computed
    Kokkos::deep_copy(force_val, particles->f);
    for (int i = 0; i < particles->N; i++) {
        for (int dir = 0; dir < dim_space; dir++) {
            Kokkos::parallel_for("check-force-condition", 1, KOKKOS_LAMBDA(const int x) {
                tmpx(i, dir) += 2 * h;
            });
            double V = -particles->evaluate_potential();
            Kokkos::parallel_for("check-force-condition", 1, KOKKOS_LAMBDA(const int x) {
                tmpx(i, dir) -= h;
            });
            V += 8 * particles->evaluate_potential();
            Kokkos::parallel_for("check-force-condition", 1, KOKKOS_LAMBDA(const int x) {
                tmpx(i, dir) -= 2 * h;
            });
            V -= 8 * particles->evaluate_potential();
            Kokkos::parallel_for("check-force-condition", 1, KOKKOS_LAMBDA(const int x) {
                tmpx(i, dir) -= h;
            });
            V += particles->evaluate_potential();
            double num_der = (V) / (12.0 * h);

            // Kokkos::parallel_for("check-force-condition", 1, KOKKOS_LAMBDA(const int x) {
            //     tmpx(i, dir) += h;
            // });
            // double V = particles->compute_potential();
            // Kokkos::parallel_for("check-force-condition", 1, KOKKOS_LAMBDA(const int x) {
            //     tmpx(i, dir) -= 2 * h;
            // });
            // V -= particles->compute_potential();
            // double num_der = (V) / (2.0 * h);

            double diff = num_der - force_val(i, dir);
            if (fabs(num_der) > 1e-6) diff /= num_der;
            diff = fabs(diff);
            if (diff > 1e-3) {
                printf("error: numerical derivative does not match force: x=%-6d dir=%-2d ", i, dir);
                printf("num_der= %-18.12g force= %-18.12g diff= %-18.12g ratio= %-18.12g \n",
                    num_der, force_val(i, dir), num_der - force_val(i, dir), num_der / force_val(i, dir));
                count++;
            }
            // restore position
            Kokkos::parallel_for("check-force-condition", 1, KOKKOS_LAMBDA(const int x) {
                tmpx(i, dir) += 2 * h;
            });
            Kokkos::fence();
        }
    }
    if (count > 0) {
        std::string s = "comparing force with numerical deriv  algorithm: " + particles->algorithm;
        add_error(errors, s);
    }
    else { printf("test passed\n"); }

}

int main(int argc, char** argv) {

    printf("chemHMC git commit %s\n", kGitHash);

    // starting kokkos
    Kokkos::initialize(argc, argv); {

        YAML::Node doc = read_params(argc, argv);
        params_class params(doc,false);
        particles_type* particles1, * particles2;

        doc["particles"]["algorithm"] = "all_neighbour_inner_parallel";
        particles1 = new particles_instance(doc, params);
        doc["particles"]["algorithm"] = "AMIC";
        particles2 = new particles_instance(doc,params);
        
        //// init the positions
        particles1->InitX(params);
        particles2->InitX(params);

        int sum = 0;
        type_x x1 = particles1->x;
        type_x x2 = particles2->x;
        // check that the initial position is the same
        int N = particles1->N;
        Kokkos::parallel_reduce("check-initial-condition", N, KOKKOS_LAMBDA(const int i, int& update) {
            double  r = Kokkos::sqrt((x1(i, 0) - x2(i, 0)) * (x1(i, 0) - x2(i, 0)) +
                (x1(i, 1) - x2(i, 1)) * (x1(i, 1) - x2(i, 1)) +
                (x1(i, 2) - x2(i, 2)) * (x1(i, 2) - x2(i, 2)));
            if (r > 1e-8) {
                printf("different position x1=(%g,%g%g)  x2=(%g,%g,%g)\n", x1(i, 0), x1(i, 1), x1(i, 2), x2(i, 0), x2(i, 1), x2(i, 2));
                update += 1;
            }
        }, sum);
        if (sum > 0)Kokkos::abort("Initial position do not match");
        else printf("the initial positon of the two ensambles is the same\n");
        // particles1->printx();
        printf("############################# timing potential calculation #########################################\n");
        Kokkos::Timer timer1;
        double V1 = particles1->compute_potential();
        Kokkos::fence();
        printf("time all_neighbour = %f s\n", timer1.seconds());
        Kokkos::Timer timer2;
        double V2 = particles2->compute_potential();
        Kokkos::fence();
        printf("time AMIC = %f s\n", timer2.seconds());

        std::vector<std::string> errors(0);
        if (fabs((V1 - V2) / V1) > 1e-6) {
            printf("%.12g   %.12g\n", V1, V2);
            add_error(errors, "error: the potential all_neighbour does not match AMIC");
        }
        else printf("Test passed: the potential is the same\n");
        printf("###################################################################################################\n");
        timer1.reset();
        particles1->compute_force();
        Kokkos::fence();
        printf("time force all_neighbour = %f s\n", timer1.seconds());

        timer2.reset();
        particles2->compute_force();
        Kokkos::fence();
        printf("time force AMIC= %f s\n", timer2.seconds());

        check_force(particles1, particles2, "all_neighbour against AMIC", errors);
        //////////////////////////////////////////////////////////////////////////////////////////
        timer1.reset();
        check_force_with_num_der(particles1, errors);
        printf("time for all_neighbour = %f s\n", timer1.seconds());

        timer2.reset();
        check_force_with_num_der(particles2, errors);
        printf("time for AMIC = %f s\n", timer2.seconds());
        //////////////////////////////////////////////////////////////////////////////////////////
        printf("\nerror recap:\n");
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
