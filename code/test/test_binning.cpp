#define CONTROL

#include <Kokkos_Core.hpp>
#include "global.hpp"
#include "git_version.hpp"
#include "HMC.hpp"
#include "identical_particles.hpp"

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

        doc["geometry"]["Lx"] = 1;
        doc["geometry"]["Ly"] = 1;
        doc["geometry"]["Lz"] = 1;

        doc["particles"]["name"] = "identical_particles";
        doc["particles"]["N"] = std::stoi(argv[opt1]);;
        doc["particles"]["mass"] = 0.1;
        doc["particles"]["beta"] = 0.5;
        doc["particles"]["cutoff"] = 0.2;
        doc["particles"]["eps"] = 0.1;
        doc["particles"]["sigma"] = 0.1;
        doc["particles"]["algorithm"] = "all_neighbour";

        particles_type* particles1, * particles2;

        particles1 = new identical_particles(doc);
        doc["particles"]["algorithm"] = "binning_serial";
        particles2 = new identical_particles(doc);
        doc["particles"]["algorithm"] = "quick_sort";
        particles_type* particles3 = new identical_particles(doc);
        //// init the positions
        particles1->InitX();
        particles2->InitX();
        particles3->InitX();

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

        Kokkos::Timer timer1;
        double V1 = particles1->compute_potential();
        printf("time in1 = %f s\n", timer1.seconds());
        Kokkos::Timer timer2;
        double V2 = particles2->compute_potential();
        printf("time in2 = %f s\n", timer2.seconds());
        timer2.reset();
        particles2->create_binning();
        printf("to bin2 in2 = %f s\n", timer2.seconds());
        printf("V1=%g  V2=%g\n", V1, V2);
        if (fabs(V1 - V2) > 1e-8)Kokkos::abort("error: the potential is not the same");
        else printf("Test passed\n");



        /////////////////////
        Kokkos::Timer t3b;
        particles3->create_binning();
        double tb = t3b.seconds();
        Kokkos::Timer timer3;
        double V3 = particles3->compute_potential();
        printf("total time quicksort = %gs  of which binning =%gs \n", timer3.seconds(), tb);
        printf("V1=%g  V3=%g\n", V1, V3);
        if (fabs(V1 - V3) > 1e-8)Kokkos::abort("error: the potential is not the same");
        else printf("Test passed\n");

        Kokkos::fence();


        t_permute_vector p2 = particles2->permute_vector;
        t_permute_vector p3 = particles3->permute_vector;
        t_bincount   bc2 = particles2->bincount;
        t_bincount   bc3 = particles3->bincount;
        t_binoffsets   bo2 = particles2->binoffsets;
        t_binoffsets   bo3 = particles3->binoffsets;

        int Nb = particles2->bintot;
        Kokkos::parallel_reduce("check-binning-condition", Nb, KOKKOS_LAMBDA(const int ib, int& update) {
            if (bc2(ib) != bc3(ib)) {
                printf("different count inside bin %d : %d vs %d  \n", ib, bc2(ib), bc3(ib));
                update++;
            }
            if (bo2(ib) != bo3(ib)) {
                printf("different offset of bin %d : %d vs %d  \n", ib, bo2(ib), bo3(ib));
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

        if (sum > 0)Kokkos::abort("binnings do not match");
        else printf("Test passed:  binning match\n");
    }
    Kokkos::finalize();
}
