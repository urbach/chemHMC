#include "identical_particles.hpp"
#include "binning.hpp"

void identical_particles::cutoff_binning() {
    bintot = 1;
    for (int i = 0;i < dim_space;i++) {
        nbin[i] = (params.L[i] / cutoff);
        sizebin[i] = params.L[i] / ((double)nbin[i]);
        bintot *= nbin[i];
    }

    bincount = t_bincount("bincount", bintot);
    binoffsets = t_binoffsets("t_binoffsets", bintot);
    permute_vector = t_permute_vector("permute_vector", N);

}


void identical_particles::serial_binning_init() {
    h_bincount = Kokkos::create_mirror_view(bincount);
    h_binoffsets = Kokkos::create_mirror_view(binoffsets);
    h_permute_vector = Kokkos::create_mirror_view(permute_vector);
}
void identical_particles::parallel_binning_init() {
   
    permute_vector_temp = t_prefix("permute_vector_temp", N);
    t_permute_vector& vec = permute_vector;
    Kokkos::parallel_for("init-permute-vector-quick-sort", N, KOKKOS_LAMBDA(const int i){
        vec(i) = i;
    });
}
inline
bool is_in_bin(type_x  x, int i, int bx, int by, int bz, const double sizebin[dim_space]) {
    int a = 0;
    double sx = sizebin[0];
    double sy = sizebin[1];
    double sz = sizebin[2];
    Kokkos::parallel_reduce("CheckValues", 1, KOKKOS_LAMBDA(const int&, int& lsum) {

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
                h_bincount(ib) = 0;
                h_binoffsets(ib) = count;
                for (int i = 0; i < N;i++) {
                    bool in = is_in_bin(x, i, bx, by, bz, sizebin);
                    if (in) {

                        h_permute_vector(count) = i;
                        h_bincount(ib) = h_bincount(ib) + 1;
                        count++;
                    }
                }

            }
        }
        Kokkos::deep_copy(bincount, h_bincount);
        Kokkos::deep_copy(binoffsets, h_binoffsets);
        Kokkos::deep_copy(permute_vector, h_permute_vector);
    }
}



void identical_particles::parallel_binning() {

    Kokkos::parallel_for("parallel-binncount", Kokkos::TeamPolicy<functor_count_bin::Tag_count_bin>(bintot, Kokkos::AUTO),
        functor_count_bin(N, nbin, sizebin, x, permute_vector_temp, permute_vector, bincount, binoffsets));
    // printf("time to bincount %gs\n", t2.seconds());
    // Kokkos::Timer t3;
    Kokkos::parallel_scan("parallel-binoffsets", Kokkos::RangePolicy<functor_count_bin::Tag_offset>(0, bintot),
        functor_count_bin(N, nbin, sizebin, x, permute_vector_temp, permute_vector, bincount, binoffsets));
    // printf("time to binoffset %gs\n", t3.seconds());
    Kokkos::parallel_for("binset", Kokkos::TeamPolicy<functor_count_bin::Tag_set_bin>(bintot, Kokkos::AUTO),
        functor_count_bin(N, nbin, sizebin, x, permute_vector_temp, permute_vector, bincount, binoffsets));

    Kokkos::parallel_for("parallel-copy-range", Kokkos::RangePolicy(0, N), copy_range(permute_vector, permute_vector_temp));
};
