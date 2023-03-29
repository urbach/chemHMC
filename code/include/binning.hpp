#ifndef binning_H
#define binning_H
#include <Kokkos_Core.hpp>
#include "global.hpp"
#include "particles.hpp"

class functor_count_bin {
public:
    struct Tag_count_bin {};
    struct Tag_offset {};
    struct Tag_set_bin {};
    const int N;
    const int nbin[dim_space];
    const double sizebin[dim_space];
    type_x x;
    t_permute_vector permute_vector_temp;
    t_permute_vector permute_vector;
    t_bincount bincount;
    t_binoffsets binoffsets;

    functor_count_bin(int N_, int nbin_[dim_space], double sizebin_[dim_space], type_x& x_,
        t_permute_vector& permute_vector_temp_, t_permute_vector& permute_vector_, t_bincount& bincount_, t_binoffsets& binoffsets_);

    KOKKOS_INLINE_FUNCTION int ctolex(const int bx, const int by, const int bz)  const {
        return bx + nbin[0] * (by + bz * nbin[1]);
    };
    KOKKOS_INLINE_FUNCTION int which_bin(const int i) const {
        const int bx = floor(x(i, 0) / sizebin[0]);
        const int by = floor(x(i, 1) / sizebin[1]);
        const int bz = floor(x(i, 2) / sizebin[2]);
        // printf("i=%d --> %d %d %d  ---> %d\n", i, bx, by, bz, ctolex(bx, by, bz));
        return ctolex(bx, by, bz);
    };
    KOKKOS_FUNCTION void operator() (Tag_count_bin, const Kokkos::TeamPolicy<>::member_type& teamMember) const ;

    KOKKOS_FUNCTION void operator() (Tag_offset, const int j, int& update, bool final) const;

    KOKKOS_FUNCTION void operator() (Tag_set_bin, const Kokkos::TeamPolicy<>::member_type& teamMember) const;

};


class copy_range {
public:
    t_permute_vector left;
    t_permute_vector right;

    copy_range(t_permute_vector& left_, t_permute_vector& right_):
        left(left_),
        right(right_) {
    };

    KOKKOS_FUNCTION void operator() (const int i) const {
        left(i) = right(i);
        // printf("%d  %d\n", i, right(i));
    };
};


#endif