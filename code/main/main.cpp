#define CONTROL

#include <Kokkos_Core.hpp>
#include "read_infile.hpp"
#include "global.hpp"
#include "git_version.h"

int main(int argc, char** argv) {

    printf("chemHMC git commit %s\n", kGitHash);

    params_class params(argc, argv);
    printf("volume: %f  %f  %f\n", params.L[0], params.L[1], params.L[2] );
    printf("Nathoms: %d\n", params.Nathoms );
}