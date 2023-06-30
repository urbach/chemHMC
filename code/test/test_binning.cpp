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
    printf("comparing binning %s\n", comparison.c_str());
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


    }, sum);
    Kokkos::fence();

    if (sum > 0)    add_error(errors, "comparing binning" + comparison);
    else printf("Test passed:  binning match\n");

}


int main(int argc, char** argv) {

    printf("chemHMC git commit %s\n", kGitHash);

    // starting kokkos
    Kokkos::initialize(argc, argv); {

        int opt1 = -1;
        for (int i = 0; i < argc; ++i) {
            if (std::strcmp(argv[i], "-N") == 0) {
                opt1 = i + 1;
            }
        }
        if (opt1 < 0) {
            std::cout << "No input file specified, Aborting" << std::endl;
            std::cout << "usage:  ./main -N ${number-of-particles}" << std::endl;
            Kokkos::abort("");
        }
        YAML::Node doc;
        doc["StartCondition"] = "hot";
        doc["seed"] = 123;

        doc["geometry"]["Lx"] = 1.1;
        doc["geometry"]["Ly"] = 1.2;
        doc["geometry"]["Lz"] = 1.3;

        doc["particles"]["name"] = "identical_particles";
        doc["particles"]["N"] = std::stoi(argv[opt1]);;
        doc["particles"]["mass"] = 0.1;
        doc["particles"]["beta"] = 0.5;
        doc["particles"]["cutoff"] = 0.4;
        doc["particles"]["eps"] = 0.1;
        doc["particles"]["sigma"] = 0.1;
        doc["particles"]["name_xyz"] = "none";
        doc["particles"]["algorithm"] = "all_neighbour";
        doc["append"] = "false";
        doc["rng_host_state"] = "tmp1";
        doc["rng_device_state"] = "tmp2";
        doc["output_file"] = "tmp3";

        particles_type* particles1, * particles2;

        particles1 = new identical_particles(doc);
        doc["particles"]["algorithm"] = "binning_serial";
        doc["rng_host_state"] = "tmp4";
        doc["rng_device_state"] = "tmp5";
        doc["output_file"] = "tmp6";
        particles2 = new identical_particles(doc);
        doc["particles"]["algorithm"] = "quick_sort";
        doc["rng_host_state"] = "tmp7";
        doc["rng_device_state"] = "tmp8";
        doc["output_file"] = "tmp9";
        particles_type* particles3 = new identical_particles(doc);
        doc["particles"]["algorithm"] = "parallel_binning";
        doc["rng_host_state"] = "tmp10";
        doc["rng_device_state"] = "tmp11";
        doc["output_file"] = "tmp12";
        particles_type* particles4 = new identical_particles(doc);

        //// init the positions
        particles1->InitX();
        particles2->InitX();
        particles3->InitX();
        particles4->InitX();

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
        printf("time binning_serial = %f s\n", timer2.seconds());
        Kokkos::Timer timer3;
        // double V3 = particles3->compute_potential();
        // Kokkos::fence();
        // printf("total time quick_sort = %gs   \n", timer3.seconds());
        Kokkos::Timer timer4;
        double V4 = particles4->compute_potential();
        Kokkos::fence();

        std::vector<std::string> errors(0);
        printf("total time parallel_binning = %gs   \n", timer4.seconds());
        if (fabs((V1 - V2) / V1) > 1e-6) {
            printf("%.12g   %.12g\n", V1, V2);
            add_error(errors, "error: the potential all_neighbour does not match binning_serial");
        }
        else printf("Test passed: the potential is the same\n");
        // if (fabs((V1 - V3) / V1) > 1e-6) {
        //     printf("%.12g   %.12g\n", V1, V3);
        //     add_error(errors, "error: the potential all_neighbour does not match quick_sort");
        // }
        // else printf("Test passed: the potential is the same\n");
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
        // Kokkos::Timer t3b;
        // particles3->create_binning();
        // Kokkos::fence();
        // printf("time to bin quick_sort  %gs\n", t3b.seconds());
        // Kokkos::Timer t3bb;
        // particles3->create_binning();
        // Kokkos::fence();
        // printf("time to bin quick_sort  %gs\n", t3bb.seconds());
        ////////////////////////////
        Kokkos::Timer t4b;
        particles4->create_binning();
        Kokkos::fence();
        printf("time to bin parallel_sort  %gs\n", t4b.seconds());
        Kokkos::fence();

        check_binning(particles2, particles4, "binning_serial  agains parallel_binning", errors);
        // check_binning(particles2, particles3, "binning_serial  agains quick_sort", errors);

        /////////////////////////////////////////////////////////////////////////////////////////
        timer1.reset();
        particles1->compute_force();
        Kokkos::fence();
        printf("time force all_neighbour = %f s\n", timer1.seconds());

        timer4.reset();
        particles4->compute_force();
        Kokkos::fence();
        printf("time force parallel_binning = %f s\n", timer4.seconds());

        check_force(particles1, particles4, "all_neighbour  agains parallel_binning", errors);
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
