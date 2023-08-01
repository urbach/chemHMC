#define CONTROL

#include <Kokkos_Core.hpp>
#include "git_version.hpp"
#include "HMC.hpp"

int main(int argc, char** argv) {

    printf("chemHMC git commit %s\n", kGitHash);

    // starting kokkos
    Kokkos::initialize(argc, argv); {
        Kokkos::Timer timer;

        HMC_class HMC;
        HMC.init(argc, argv);
        
        HMC.measure();

        printf("total kokkos time = %f s\n", timer.seconds());
    }
    Kokkos::finalize();
}
