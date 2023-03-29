#include "binning.hpp"

functor_count_bin::functor_count_bin(int N_, int nbin_[dim_space], double sizebin_[dim_space], type_x& x_,
    t_permute_vector& permute_vector_temp_, t_permute_vector& permute_vector_, t_bincount& bincount_, t_binoffsets& binoffsets_):
    N(N_),
    nbin{ nbin_[0], nbin_[1], nbin_[2] },
    sizebin{ sizebin_[0], sizebin_[1], sizebin_[2] },
    x(x_),
    permute_vector_temp(permute_vector_temp_),
    permute_vector(permute_vector_),
    bincount(bincount_),
    binoffsets(binoffsets_) {
};

KOKKOS_FUNCTION void functor_count_bin::operator() (Tag_count_bin, const Kokkos::TeamPolicy<>::member_type& teamMember) const {
    const int ib = teamMember.league_rank();
    bincount(ib) = 0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, N), [=](const int i, int& inBinUpdate) {
        const int b = which_bin(permute_vector(i));
        if (b == ib) {
            inBinUpdate++;
        }
        }, bincount(ib));
};

KOKKOS_FUNCTION void functor_count_bin::operator() (Tag_offset, const int j, int& update, bool final) const {
    const int vec_j = bincount(j);
    if (final) {
        binoffsets(j) = update; // only update array on final pass
    }
    // For exclusive scan (0,...), change the update value after
    // updating array, like we do here. For inclusive scan (1,...),
    // change the update value before updating array.
    update += vec_j;
};

KOKKOS_FUNCTION void functor_count_bin::operator() (Tag_set_bin, const Kokkos::TeamPolicy<>::member_type& teamMember) const {
    const int ib = teamMember.league_rank();
    //  Kokkos::single(Kokkos::PerTeam(teamMember), [&]() {
    //     printf("bin %d  binncount %d binoff %d\n",ib,bincount(ib), binoffsets(ib));
    //  });
    Kokkos::parallel_scan(Kokkos::TeamThreadRange(teamMember, N), [&](const int i, int& update, bool final) {
        const int b = which_bin(permute_vector(i));
        if (b == ib) {
            if (final) {
                permute_vector_temp(update + binoffsets(ib)) = permute_vector(i);
                // printf("bin %d: %d <--- %d \n", ib, update + binoffsets(i), i);
            }
            update++;
        }
        }
    );

}