#include "identical_particles.hpp"

void identical_particles::quick_sort_init() {
    before = t_bool("before", N);
    after = t_bool("afater", N);
    sb = t_prefix("sb", N);
    sa = t_prefix("sa", N);
    permute_vector_temp = t_prefix("permute_vector_temp", N);
    t_permute_vector& vec = permute_vector;
    Kokkos::parallel_for("init-permute-vector-quick-sort", N, KOKKOS_LAMBDA(const int i){
        vec(i) = i;
    });
}



// A utility function to swap two elements
// void swap(int i, int j) {
//     int t = permute_vector(i);
//     permute_vector(i) = permute_vector(j);
//     permute_vector(j) = t;
// }



class functor_set_zero {
public:
    t_bool before;
    t_bool after;

    functor_set_zero(t_bool& before_, t_bool& after_):
        before(before_),
        after(after_) {
    };

    KOKKOS_FUNCTION void operator() (const int j) const {
        before(j) = 0;
        after(j) = 0;
    };
};

class quicksort_compare {
public:
    struct Tag_all {};
    struct Tag_no_pivot {};
    const int pi;
    const int nbin[dim_space];
    const double sizebin[dim_space];
    type_x x;
    t_permute_vector permute_vector;
    t_bool before;
    t_bool after;


    quicksort_compare(int& pi_, int nbin_[dim_space], double sizebin_[dim_space], type_x& x_,
        t_permute_vector& permute_vector_, t_bool& before_, t_bool& after_):
        pi(pi_),
        nbin{ nbin_[0], nbin_[1], nbin_[2] },
        sizebin{ sizebin_[0], sizebin_[1], sizebin_[2] },
        x(x_),
        permute_vector(permute_vector_),
        before(before_),
        after(after_) {
    };
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
    KOKKOS_FUNCTION void operator() (struct Tag_all, const int j, int& update) const {
        // printf("permute %d  %d: %d  %d\n", j, pi, permute_vector(j), permute_vector(pi));
        const int a = which_bin(permute_vector(j));
        const int b = which_bin(permute_vector(pi));
        // printf("a=%d b=%d\n", a, b);
        if (a <= b) {
            before(j) = 1;
            after(j) = 0;
            update++;
        }
        else {
            after(j) = 1;
            before(j) = 0;
        }
        // printf("end a=%d b=%d\n", a, b);
    };

    KOKKOS_FUNCTION void operator() (struct Tag_no_pivot, const int j, int& update) const {
        if (j != pi) {
            const int a = which_bin(permute_vector(j));
            const int b = which_bin(permute_vector(pi));
            // printf("a=%d b=%d\n", a, b);
            if (a <= b) {
                before(j) = 1;
                after(j) = 0;
                update++;
            }
            else {
                after(j) = 1;
                before(j) = 0;
            }
        }
        else {
            before(j) = 0;
            after(j) = 0;
        }

    };
};




class functor_count_bin {
public:
    struct Tag_all {};
    struct Tag_no_pivot {};
    const int N;
    const int nbin[dim_space];
    const double sizebin[dim_space];
    type_x x;
    t_permute_vector permute_vector;
    t_bincount bincount;
    t_binoffsets binoffsets;

    functor_count_bin(int N_, int nbin_[dim_space], double sizebin_[dim_space], type_x& x_,
        t_permute_vector& permute_vector_, t_bincount& bincount_, t_binoffsets& binoffsets_):
        N(N_),
        nbin{ nbin_[0], nbin_[1], nbin_[2] },
        sizebin{ sizebin_[0], sizebin_[1], sizebin_[2] },
        x(x_),
        permute_vector(permute_vector_),
        bincount(bincount_),
        binoffsets(binoffsets_) {
    };
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
    KOKKOS_FUNCTION void operator() (const Kokkos::TeamPolicy<>::member_type& teamMember) const {
        const int ib = teamMember.league_rank();
        bincount(ib) = 0;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, N), [=](const int i, int& inBinUpdate) {
            const int b = which_bin(permute_vector(i));
            if (b == ib) {
                inBinUpdate++;
            }
            }, bincount(ib));

    };

    KOKKOS_FUNCTION void operator() (const int j, int& update, bool final) const {
        const int vec_j = bincount(j);
        if (final) {
            binoffsets(j) = update; // only update array on final pass
        }
        // For exclusive scan (0,...), change the update value after
        // updating array, like we do here. For inclusive scan (1,...),
        // change the update value before updating array.
        update += vec_j;
    };
};


namespace arrayN2 {  // namespace helps with name resolution in reduction identity 
    template< class ScalarType, int N >
    struct array_type {
        ScalarType the_array[N];

        KOKKOS_INLINE_FUNCTION   // Default constructor - Initialize to 0's
            array_type() {
            for (int i = 0; i < N; i++) { the_array[i] = 0; }
        }
        KOKKOS_INLINE_FUNCTION   // Copy Constructor
            array_type(const array_type& rhs) {
            for (int i = 0; i < N; i++) {
                the_array[i] = rhs.the_array[i];
            }
        }
        KOKKOS_INLINE_FUNCTION   // add operator
            array_type& operator += (const array_type& src) {
            for (int i = 0; i < N; i++) {
                the_array[i] += src.the_array[i];
            }
            return *this;
        }
    };
    typedef array_type<int, 2> ValueType;  // used to simplify code below
}
namespace Kokkos { //reduction identity must be defined in Kokkos namespace
    template<>
    struct reduction_identity< arrayN2::ValueType > {
        KOKKOS_FORCEINLINE_FUNCTION static arrayN2::ValueType sum() {
            return arrayN2::ValueType();
        }
    };
}

class functor_prefix_and_set {
public:
    const int  offset_x;
    // const int pi; // position of the pivot
    const int pi_pos; // where to put the pivot
    t_bool before;
    t_bool after;
    t_permute_vector permute_vector_temp;
    t_permute_vector permute_vector;
    functor_prefix_and_set(t_permute_vector& permute_vector_temp_, t_permute_vector& permute_vector_, t_bool before_, t_bool after_, int  offset_x_, int pi_pos_ /*, int pi_*/):
        offset_x(offset_x_),
        // pi(pi_),
        pi_pos(pi_pos_),
        before(before_),
        after(after_),
        permute_vector_temp(permute_vector_temp_),
        permute_vector(permute_vector_) {
    };
    KOKKOS_FUNCTION void operator() (const int j, arrayN2::ValueType& update, const bool final) const {
        // Load old value in case we update it before accumulating
        int b_j = before(j);
        int a_j = after(j);

        if (before(j) == 1) {
            if (final) {
                b_j = update.the_array[0];// only update array on final pass
                permute_vector_temp(b_j + offset_x) = permute_vector(j);
            }
        }
        else if (after(j) == 1) {
            if (final) {
                a_j = update.the_array[1];// only update array on final pass
                permute_vector_temp(a_j + pi_pos + 1) = permute_vector(j);
            }
        }

        // For exclusive scan (0,...), change the update value after
        // updating array, like we do here. For inclusive scan (1,...),
        // change the update value before updating array.
        update.the_array[0] += b_j;
        update.the_array[1] += a_j;
    };
};


class functor_prefix {
public:
    t_prefix vec;
    t_bool vecp;
    functor_prefix(t_prefix& vec_, t_bool& vecp_): vec(vec_), vecp(vecp_) {};
    KOKKOS_FUNCTION void operator() (const int j, double& update, const bool final) const {
        // Load old value in case we update it before accumulating
        const int vec_j = vecp(j);
        if (final) {
            vec(j) = update; // only update array on final pass
        }
        // For exclusive scan (0,...), change the update value after
        // updating array, like we do here. For inclusive scan (1,...),
        // change the update value before updating array.
        update += vec_j;
    };
};


class functor_set_sorted {
public:
    const int  offset_x;
    const int  offset_v;
    t_prefix vec;
    t_permute_vector permute_vector;
    t_permute_vector permute_vector_temp;
    t_bool vecif;


    functor_set_sorted(t_permute_vector& permute_vector_temp_, t_permute_vector& permute_vector_, t_prefix vec_, t_bool vecif_, int  offset_x_, int offset_v_):
        offset_x(offset_x_),
        offset_v(offset_v_),
        vec(vec_),
        permute_vector(permute_vector_),
        permute_vector_temp(permute_vector_temp_),
        vecif(vecif_) {
    };


    KOKKOS_FUNCTION void operator() (const int j) const {
        if (vecif(j) == 1)
            permute_vector_temp(vec(j + offset_v) + offset_x) = permute_vector(j);
    };

};

class functor_set_pivot {
public:
    const int was;
    t_permute_vector permute_vector;
    t_permute_vector permute_vector_temp;

    functor_set_pivot(t_permute_vector& permute_vector_temp_, t_permute_vector& permute_vector_, int  was_):
        was(was_),
        permute_vector(permute_vector_),
        permute_vector_temp(permute_vector_temp_) {
    };

    KOKKOS_FUNCTION void operator() (const int will_be) const {
        permute_vector_temp(will_be) = permute_vector(was);
    };
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
    };
};


/* This function takes last element as pivot, places
 *  the pivot element at its correct position in sorted
 *   array, and places all smaller (smaller than pivot)
 *  to left of pivot and all greater elements to right
 *  of pivot */
int identical_particles::partition_middle(int low, int high) {
    int pi = (high + low) / 2 + 1;
    int pi_pos = 0;
    Kokkos::parallel_reduce("compare_to_pivot", Kokkos::RangePolicy<quicksort_compare::Tag_no_pivot>(low, high + 1),
        quicksort_compare(pi, nbin, sizebin, x, permute_vector, before, after), pi_pos);
    pi_pos += low;
    Kokkos::parallel_scan("prefix_and_set", Kokkos::RangePolicy(low, high + 1), functor_prefix_and_set(permute_vector_temp, permute_vector, before, after, low, pi_pos));
    Kokkos::parallel_for("set-pivot", Kokkos::RangePolicy(pi_pos, pi_pos + 1), functor_set_pivot(permute_vector_temp, permute_vector, pi));

    Kokkos::parallel_for("copy-range", Kokkos::RangePolicy(low, high + 1), copy_range(permute_vector, permute_vector_temp));
    return pi_pos;
}

/* This function takes last element as pivot, places
 *  the pivot element at its correct position in sorted
 *   array, and places all smaller (smaller than pivot)
 *  to left of pivot and all greater elements to right
 *  of pivot */
int identical_particles::partition_high(int low, int high) {
    int pi = high;
    int pi_pos = 0;
    Kokkos::parallel_reduce("compare_to_pivot", Kokkos::RangePolicy<quicksort_compare::Tag_all>(low, high),
        quicksort_compare(pi, nbin, sizebin, x, permute_vector, before, after), pi_pos);
    pi_pos += low;
    Kokkos::parallel_scan("prefix_and_set", Kokkos::RangePolicy(low, high), functor_prefix_and_set(permute_vector_temp, permute_vector, before, after, low, pi_pos));
    Kokkos::parallel_for("set-pivot", Kokkos::RangePolicy(pi_pos, pi_pos + 1), functor_set_pivot(permute_vector_temp, permute_vector, pi));
    Kokkos::parallel_for("copy-range", Kokkos::RangePolicy(low, high + 1), copy_range(permute_vector, permute_vector_temp));
    return pi_pos;
}


/* This function takes last element as pivot, places
 *  the pivot element at its correct position in sorted
 *   array, and places all smaller (smaller than pivot)
 *  to left of pivot and all greater elements to right
 *  of pivot */
int identical_particles::partition(int low, int high) {
    // double pivot = arr[order[high]];    // pivot
    // int i = (low - 1);  // Index of smaller element

    // for (int j = low; j <= high - 1; j++) {
    //     // If current element is smaller than or
    //     // equal to pivot
    //     // if (arr[order[j]] <= pivot) {
    //     if (compare(j, x, high)) {
    //         i++;    // increment index of smaller element
    //         swap(i, j);
    //     }
    // }
    // swap(i + 1, high);
    int pi = high;
    int pi_pos = 0;

    Kokkos::Timer time_1;

    // Kokkos::parallel_for("set-zero", Kokkos::RangePolicy(low, high), functor_set_zero(before, after));
    // set the view before with 1 is the element should be before , set the elemnt of after to 1 if elemets are after 
    Kokkos::parallel_reduce("compare_to_pivot", Kokkos::RangePolicy<quicksort_compare::Tag_all>(low, high),
        quicksort_compare(pi, nbin, sizebin, x, permute_vector, before, after), pi_pos);
    pi_pos += low;
    Kokkos::parallel_scan("prefix_b", Kokkos::RangePolicy(low, high), functor_prefix(sb, before));
    Kokkos::parallel_scan("prefix_a", Kokkos::RangePolicy(low, high), functor_prefix(sa, after));
    // add the point that are smaller then the pivot
    Kokkos::parallel_for("set-sorted-before", Kokkos::RangePolicy(low, high),
        functor_set_sorted(permute_vector_temp, permute_vector, sb, before, low, 0));
    // add the pivot here
    Kokkos::parallel_for("set-pivot", Kokkos::RangePolicy(pi_pos, pi_pos + 1), functor_set_pivot(permute_vector_temp, permute_vector, pi));
    // add the point that are larger then the pivot
    Kokkos::parallel_for("set-sorted-after", Kokkos::RangePolicy(low, high), functor_set_sorted(permute_vector_temp, permute_vector, sa, after,
        (pi_pos + 1), 0));


    printf("time for scan 1: %gs\n", time_1.seconds());

    /// new
    // Kokkos::View<int[1]> pi_pos_d("pivot_position_device");
    pi_pos = 0;
    Kokkos::Timer time_2;
    // Kokkos::parallel_scan("before_pivot", Kokkos::RangePolicy<quicksort_functor_compare::Tag_before>(low, high),
    //     quicksort_functor_compare(pi, nbin, sizebin, x, permute_vector_temp, permute_vector, low, pi_pos_d, high));
    // Kokkos::parallel_scan("after_pivot", Kokkos::RangePolicy<quicksort_functor_compare::Tag_after>(low, high),
    //     quicksort_functor_compare(pi, nbin, sizebin, x, permute_vector_temp, permute_vector, 0, pi_pos_d, high));

    // Kokkos::parallel_for("set-zero", Kokkos::RangePolicy(low, high), functor_set_zero(before, after));
    Kokkos::parallel_reduce("compare_to_pivot", Kokkos::RangePolicy<quicksort_compare::Tag_all>(low, high),
        quicksort_compare(pi, nbin, sizebin, x, permute_vector, before, after), pi_pos);
    pi_pos += low;
    Kokkos::parallel_scan("prefix_and_set", Kokkos::RangePolicy(low, high), functor_prefix_and_set(permute_vector_temp, permute_vector, before, after, low, pi_pos));
    Kokkos::parallel_for("set-pivot", Kokkos::RangePolicy(pi_pos, pi_pos + 1), functor_set_pivot(permute_vector_temp, permute_vector, pi));

    printf("time for scan 2: %gs\n", time_2.seconds());
    // ///

    /// new
    pi = (high + low) / 2 + 1;
    pi_pos = 0;
    Kokkos::Timer time_3;
    Kokkos::parallel_reduce("compare_to_pivot", Kokkos::RangePolicy<quicksort_compare::Tag_no_pivot>(low, high + 1),
        quicksort_compare(pi, nbin, sizebin, x, permute_vector, before, after), pi_pos);
    pi_pos += low;
    printf("new pivot position %d\n", pi_pos);
    Kokkos::parallel_scan("prefix_and_set", Kokkos::RangePolicy(low, high), functor_prefix_and_set(permute_vector_temp, permute_vector, before, after, low, pi_pos));
    Kokkos::parallel_for("set-pivot", Kokkos::RangePolicy(pi_pos, pi_pos + 1), functor_set_pivot(permute_vector_temp, permute_vector, pi));

    printf("time for scan 3: %gs\n", time_3.seconds());
    // ///


    // /// new  4
    // pi = (high + low) / 2 + 1;
    // pi_pos = 0;
    // int pi_pos_add = 0;
    // Kokkos::Timer time_4;
    // Kokkos::parallel_reduce("compare_to_pivot", Kokkos::RangePolicy<quicksort_compare::Tag_all>(low, pi),
    //     quicksort_compare(pi, nbin, sizebin, x, permute_vector, before, after), pi_pos);
    // Kokkos::parallel_for("set-zero", Kokkos::RangePolicy(pi, pi + 1), functor_set_zero(before, after));
    // printf("range %d %d %d\n",pi+1,high+1,(high - low) / 2 + 1);
    // Kokkos::parallel_reduce("compare_to_pivot", Kokkos::RangePolicy<quicksort_compare::Tag_all>(pi + 1, high + 1),
    //     quicksort_compare(pi, nbin, sizebin, x, permute_vector, before, after), pi_pos_add);
    // printf("new pivot position %d  = %d +%d+%d\n", pi_pos + low + pi_pos_add, pi_pos, low, pi_pos_add);
    // pi_pos += low + pi_pos_add;
    // printf("new pivot position %d\n", pi_pos);
    // Kokkos::parallel_scan("prefix_and_set", Kokkos::RangePolicy(low, high), functor_prefix_and_set(permute_vector_temp, permute_vector, before, after, low, pi_pos));
    // Kokkos::parallel_for("set-pivot", Kokkos::RangePolicy(pi_pos, pi_pos + 1), functor_set_pivot(permute_vector_temp, permute_vector, pi));

    // printf("time for scan 4: %gs\n", time_4.seconds());
    // // ///

    Kokkos::parallel_for("copy-range", Kokkos::RangePolicy(low, high + 1), copy_range(permute_vector, permute_vector_temp));
    return pi_pos;
}

/* The main function that implements QuickSort
 * arr[] --> Array to be sorted,
 * low  --> Starting index,
 * high  --> Ending index */
void identical_particles::quickSort(int low, int high) {
    if (low < high) {

        int pi = partition_middle(low, high);

        // Separately sort elements before
        // partition and after partition
        quickSort(low, pi - 1);
        quickSort(pi + 1, high);
    }
}

void identical_particles::create_quick_sort() {
    Kokkos::Timer t1;
    quickSort(0, N - 1);
    printf("time to quicksort %gs\n", t1.seconds());
    Kokkos::Timer t2;
    Kokkos::parallel_for("quicksort-binncount", Kokkos::TeamPolicy(bintot, Kokkos::AUTO),
        functor_count_bin(N, nbin, sizebin, x, permute_vector, bincount, binoffsets));
    printf("time to bincount %gs\n", t2.seconds());
    Kokkos::Timer t3;
    Kokkos::parallel_scan("quicksort-binoffsets", bintot,
        functor_count_bin(N, nbin, sizebin, x, permute_vector, bincount, binoffsets));
    printf("time to binoffset %gs\n", t3.seconds());


};


