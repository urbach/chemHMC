#define CONTROL

#include <Kokkos_Core.hpp>
#include "git_version.h"
#include "HMC.h"

int main(int argc, char** argv) {

    printf("chemHMC git commit %s\n", kGitHash);

    // starting kokkos
    Kokkos::initialize(argc, argv); {
        Kokkos::Timer timer;

        HMC_class HMC;
        HMC.init(argc, argv);
        // init random pool

        



        printf("total kokkos time = %f s\n", timer.seconds());
    }
    Kokkos::finalize();
}