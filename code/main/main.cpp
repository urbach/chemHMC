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
        
        if (HMC.simulation_type == SimulationType::HMC) {
            printf("Starting HMC run.\n");
            HMC.run();
        } else if (HMC.simulation_type == SimulationType::MD){
            printf("Starting MD run.\n");
            HMC.run_MD();
        } else if (HMC.simulation_type == SimulationType::VolumeMoveHMC){
            printf("Starting HMC run with volume MC moves.\n");
            HMC.run_VolumeMoveHMC();
        }
        printf("total kokkos time = %f s\n", timer.seconds());
    }
    Kokkos::finalize();
}
