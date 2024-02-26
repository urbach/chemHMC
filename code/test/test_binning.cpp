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


void check_binning(particles_type* particles2, particles_type* particles3, std::string comparison, std::vector<std::string>& errors) {
    t_permute_vector p2 = particles2->permute_vector;
    t_permute_vector p3 = particles3->permute_vector;
    t_bincount   bc2 = particles2->bincount;
    t_bincount   bc3 = particles3->bincount;
    t_binoffsets   bo2 = particles2->binoffsets;
    t_binoffsets   bo3 = particles3->binoffsets;
    printf("########################################################################################\n");
    printf("comparing binning  %s\n", comparison.c_str());
    int Nb = particles2->bintot;
    int sum = 0;
    int nbin[3] = { particles2->nbin[0],particles2->nbin[1],particles2->nbin[2] };
    Kokkos::parallel_reduce("check-binning-condition", Nb, KOKKOS_LAMBDA(const int ib, int& update) {
        int bx, by, bz;
        // we need this manually here we can not call lextoc from here
        bz = ib / (nbin[0] * nbin[1]);
        by = (ib - bz * nbin[0] * nbin[1]) / (nbin[0]);
        bx = ib - nbin[0] * (by + bz * nbin[1]);
        //// 
        if (bc2(ib) != bc3(ib)) {
            printf("different count inside bin %d=(%d, %d, %d) : %d vs %d  \n", ib, bx, by, bz, bc2(ib), bc3(ib));
            update++;
        }
        if (bo2(ib) != bo3(ib)) {
            printf("different offset of bin %d=(%d, %d, %d) : %d vs %d  \n", ib, bx, by, bz, bo2(ib), bo3(ib));
            update++;
        }
        if (bc2(ib) >= bc3(ib)) {
            for (int i = 0;i < bc2(ib);i++) {
                int found = 1;
                for (int j = 0;j < bc3(ib);j++) {
                    if (p2(i + bo2(ib)) == p3(j + bo3(ib)))
                        found = 0;
                }
                if (found == 1) {
                    printf("Particle %d inside bin %d not found in the second partition  \n", p2(i + bo2(ib)), ib);
                    update++;
                }
            }
        }
        else {
            for (int i = 0;i < bc3(ib);i++) {
                int found = 1;
                for (int j = 0;j < bc2(ib);j++) {
                    if (p2(j + bo2(ib)) == p3(i + bo3(ib)))
                        found = 0;
                }
                if (found == 1) {
                    printf("Particle %d inside bin %d not found in the first partition  \n", p3(i + bo3(ib)), ib);
                    update++;
                }
            }
        }


    }, sum);
    Kokkos::fence();

    if (sum > 0)    add_error(errors, "comparing binning" + comparison);
    else printf("Test passed:  binning match\n");

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
        particles1 = new identical_particles(doc, params);

        doc["particles"]["algorithm"] = "binning_serial";
        particles2 = new identical_particles(doc, params);
        doc["particles"]["algorithm"] = "quick_sort";
        // doc["rng_host_state"] = "tmp7";
        // doc["rng_device_state"] = "tmp8";
        // doc["output_file"] = "tmp9";
        particles_type* particles3 = new identical_particles(doc, params);
        doc["particles"]["algorithm"] = "parallel_binning";
        particles_type* particles4 = new identical_particles(doc, params);

        //// init the positions
        particles1->InitX(params);
        particles2->InitX(params);
        particles3->InitX(params);
        particles4->InitX(params);

        int sum = 0;
        type_x x1 = particles1->x;
        type_x x2 = particles4->x;
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
        particles2->create_binning();
        double V2 = particles2->compute_potential();
        Kokkos::fence();
        printf("time binning_serial = %f s\n", timer2.seconds());
        Kokkos::Timer timer3;
        double V3 = particles3->compute_potential();
        Kokkos::fence();
        printf("total time quick_sort = %gs   \n", timer3.seconds());
        Kokkos::Timer timer4;
        particles4->create_binning();
        double V4 = particles4->compute_potential();
        Kokkos::fence();

        std::vector<std::string> errors(0);
        printf("total time parallel_binning = %gs   \n", timer4.seconds());
        if (fabs((V1 - V2) / V1) > 1e-6) {
            printf("%.12g   %.12g\n", V1, V2);
            add_error(errors, "error: the potential all_neighbour does not match binning_serial");
        }
        else printf("Test passed: the potential is the same\n");
        if (fabs((V1 - V3) / V1) > 1e-6) {
            printf("%.12g   %.12g\n", V1, V3);
            add_error(errors, "error: the potential all_neighbour does not match quick_sort");
        }
        else printf("Test passed: the potential is the same\n");
        if (fabs((V1 - V4) / V1) > 1e-6) {
            printf("%.12g   %.12g\n", V1, V4);
            add_error(errors, "error: the potential all_neighbour does not match parallel_binning");
        }
        else printf("Test passed: the potential is the same\n");
        printf("###################################################################################################\n");

        timer2.reset();
        particles2->create_binning();
        Kokkos::fence();
        printf("time to bin  serial = %g s\n", timer2.seconds());
        /////////////////////
        Kokkos::Timer t3b;
        particles3->create_binning();
        Kokkos::fence();
        printf("time to bin quick_sort  %gs\n", t3b.seconds());
        Kokkos::Timer t3bb;
        particles3->create_binning();
        Kokkos::fence();
        printf("time to bin quick_sort second time: %gs\n", t3bb.seconds());
        ////////////////////////////
        Kokkos::Timer t4b;
        particles4->create_binning();
        Kokkos::fence();
        printf("time to bin parallel_sort  %gs\n", t4b.seconds());
        Kokkos::fence();

        check_binning(particles2, particles4, "binning_serial  agains parallel_binning", errors);
        check_binning(particles2, particles3, "binning_serial  agains quick_sort", errors);

        /////////////////////////////////////////////////////////////////////////////////////////
        printf("###################################################################################################\n");
        timer1.reset();
        particles1->compute_force();
        Kokkos::fence();
        printf("time force all_neighbour = %f s\n", timer1.seconds());

        timer2.reset();
        particles2->compute_force();
        Kokkos::fence();
        printf("time force all_neighbour = %f s\n", timer2.seconds());

        timer3.reset();
        particles3->compute_force();
        Kokkos::fence();
        printf("time force quick_sort = %f s\n", timer3.seconds());

        timer4.reset();
        particles4->compute_force();
        Kokkos::fence();
        printf("time force parallel_binning = %f s\n", timer4.seconds());

        check_force(particles1, particles2, "all_neighbour  agains serial_binning", errors);
        check_force(particles1, particles3, "all_neighbour  agains quick_sort", errors);
        check_force(particles1, particles4, "all_neighbour  agains parallel_binning", errors);
        //////////////////////////////////////////////////////////////////////////////////////////

        check_force_with_num_der(particles1, errors);
        // check_force_with_num_der(particles2, errors);
        check_force_with_num_der(particles4, errors);

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
