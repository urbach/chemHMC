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
        HMC.init(argc, argv, false);
        
        HMC.measure();

        printf("total kokkos time = %f s\n", timer.seconds());
    }
    Kokkos::finalize();
}

/* void HMC_class::measure() {
    auto& p = integrator->particles;
    FILE* file = NULL;
    file = fopen(params.nameout.c_str(), "r");
    if (file == NULL) {
        printf("error in opening file %s\n", params.nameout.c_str());
        Kokkos::abort("abort");
    }
    int confs = p->how_many_confs_xyz(file);
    printf("the input file contains %d configurations \n", confs);
    ////////////////////
    FILE* file_RDF = NULL;
    file_RDF = fopen(p->filename_RDF.c_str(), "w");
    if (file_RDF == NULL) {
        printf("error in opening file %s\n", p->filename_RDF.c_str());
        Kokkos::abort("abort");
    }
    p->write_header_RDF(file_RDF, confs);

    ///////////////////////
    for (int i = 0; i < confs; i++) {
        p->read_next_confs_xyz(file);
        p->compute_RDF();
        p->write_RDF(file_RDF, i);
    }
    fclose(file_RDF);
    fclose(file);
} */