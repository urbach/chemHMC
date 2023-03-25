#include "identical_particles.hpp"

void identical_particles::cutoff_binning() {
    bintot=1;
    for (int i = 0;i < dim_space;i++) {
        nbin[i] = (params.L[i] / cutoff);
        sizebin[i] = params.L[i] / ((double)nbin[i]);
        bintot*=nbin[i];
    }

    bincount = t_bincount("bincount", bintot);
    binoffsets = t_binoffsets("t_binoffsets", bintot);
    permute_vector = t_permute_vector("permute_vector", N);
    h_bincount = Kokkos::create_mirror_view(bincount);
    h_binoffsets = Kokkos::create_mirror_view(binoffsets);
    h_permute_vector = Kokkos::create_mirror_view(permute_vector);
    printf("permute_ vector: %d  %d\n", permute_vector.is_allocated(), permute_vector.is_hostspace);
    printf("h_permute_ vector: %d %d\n", h_permute_vector.is_allocated(), h_permute_vector.is_hostspace);
}



inline
bool is_in_bin(type_x  x, int i, int bx, int by, int bz, const double sizebin[dim_space]) {
    int a = 0;
    double sx = sizebin[0];
    double sy = sizebin[1];
    double sz = sizebin[2];
    Kokkos::parallel_reduce("CheckValues", 1, KOKKOS_LAMBDA(const int&, int& lsum) {
        // printf("x=%g %g %g\n", x(i, 0), x(i, 1), x(i, 2));
        // printf("bin=[%g,%g]   [%g,%g]   [%g,%g]\n", bz* sx, (bz + 1)* sx, by* sy, (by + 1)* sy, bx* sz, (bx + 1)* sz);
        if (x(i, 2) >= bz * sx && x(i, 2) < (bz + 1) * sx &&
            x(i, 1) >= by * sy && x(i, 1) < (by + 1) * sy &&
            x(i, 0) >= bx * sz && x(i, 0) < (bx + 1) * sz)
            lsum++;
    }, a);
    Kokkos::fence();
    if (a > 0) {
        return true;
    }
    else
        return false;
    return false;
}

void identical_particles::serial_binning() {
    int count = 0;
    int N = x.extent(0);
    for (int bz = 0; bz < nbin[2]; bz++) {
        for (int by = 0; by < nbin[1]; by++) {
            for (int bx = 0; bx < nbin[0]; bx++) {
                int ib = ctolex(bx, by, bz);
                // printf("ib=%d = %d %d %d  nbin=%d %d %d  bintot=%d\n", ib, bx, by, bz, nbin[0], nbin[1], nbin[2], bintot);
                h_bincount(ib) = 0;
                h_binoffsets(ib) = count;
                for (int i = 0; i < N;i++) {
                    bool in = is_in_bin(x, i, bx, by, bz, sizebin);
                    if (in) {
                        // printf("particle %d is in   %d\n", i, h_permute_vector.size());
                        h_permute_vector(count) = i;
                        h_bincount(ib) = h_bincount(ib) + 1;
                        count++;
                    }
                }
                // Kokkos::parallel_reduce("identical_particles-LJ-potential", Kokkos::RangePolicy(0, N), 
                //  KOKKOS_LAMBDA (const int i, double& update) {
                //     is_in_bin(x, i, bx, by, bz, sizebin);
                //     update++;
                //  }, bincount(ib));
                //  count+=bincount(ib);
            }

        }
    }
    Kokkos::deep_copy(bincount, h_bincount);
    Kokkos::deep_copy(binoffsets, h_binoffsets);
    Kokkos::deep_copy(permute_vector, h_permute_vector);
    // for (int i = 0;i < h_bincount.size();i++) {
    //     printf("bin %d  size %d offset %d\n", i, h_bincount(i), h_binoffsets(i));
    //     for (int j = 0; j < h_bincount(i);j++)
    //         printf("%d\t", h_permute_vector(j + h_binoffsets(i)));
    //     printf("\n");
    // }

}


// void binning::create_binning(type_x  x) {
//     int count = 0;
//     int N = x.extent(0);
//     // for (int ib = 0; ib < bintot; ib++) {
//     Kokkos::parallel_for("set-bincount", team_policy(N, Kokkos::AUTO), KOKKOS_LAMBDA(const int& ib){
//         bincount(ib) = 0;
//     });

// for (int i = 0; i < N;i++) {
//     int bx = floor(x(i, 0) / sizebin[0]);
//     int by = floor(x(i, 1) / sizebin[1]);
//     int bz = floor(x(i, 2) / sizebin[2]);
//     bincount(ctolex(bx, by, bz))++;
// }
// for (int ib = 1; ib < bintot; ib++) {
//     binoffsets(ib) = binoffsets(ib - 1) + bincount(ib);
// }
// for (int i = 0; i < N;i++) {
//     int ib = which_bin(x, i);
//     int binncounter = 0;
//     permute_vector(binoffsets(ib) + binncounter) = i;
// }
// }
