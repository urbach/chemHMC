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
        int opt2 = -1;
        for (int i = 0; i < argc; ++i) {
            if (std::strcmp(argv[i], "-i1") == 0) {
                opt1 = i + 1;
            }
            if (std::strcmp(argv[i], "-i2") == 0) {
                opt2 = i + 1;
            }
        }
        if (opt1 * opt2 < 0) {
            std::cout << "No input file specified, Aborting" << std::endl;
            std::cout << "usage:  ./main -i1 infile1.in -i2 infile2.in" << std::endl;
            exit(1);
        }
        YAML::Node doc1, doc2;
        particles_type* particles1, * particles2;
        std::string infilename1 = argv[opt1], infilename2 = argv[opt2];

        doc1 = YAML::LoadFile(infilename1);
        doc2 = YAML::LoadFile(infilename2);

        particles1 = new identical_particles(doc1);
        particles2 = new identical_particles(doc2);

        particles1->InitX();
        particles2->InitX();
        int sum = 0;
        type_x x1 = particles1->x;
        type_x x2 = particles2->x;
        printf("N=%d\n", particles1->N);
        int N = particles1->N;
        Kokkos::parallel_reduce("chech-initial-condition", N,     KOKKOS_LAMBDA (const int i, int& update) {
            double  r = Kokkos::sqrt((x1(i, 0) - x2(i, 0)) * (x1(i, 0) - x2(i, 0)) +
                (x1(i, 1) - x2(i, 1)) * (x1(i, 1) - x2(i, 1)) +
                (x1(i, 2) - x2(i, 2)) * (x1(i, 2) - x2(i, 2)));
            if (r > 1e-8) {
                printf("different position x1=(%g,%g%g)  x2=(%g,%g,%g)\n", x1(i, 0), x1(i, 1), x1(i, 2), x2(i, 0), x2(i, 1), x2(i, 2));
                update += 1;
            }
        }, sum);
        if (sum > 0)Kokkos::abort("Initial position do not match");

        Kokkos::Timer timer1;
        double V1 = particles1->compute_potential();
        printf("time in1 = %f s\n", timer1.seconds());
        Kokkos::Timer timer2;
        double V2 = particles2->compute_potential();
        printf("time in2 = %f s\n", timer2.seconds());

        printf("V1=%g  V2=%g\n", V1, V2);

    }
    Kokkos::finalize();
}
