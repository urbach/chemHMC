#include "identical_particles.hpp"
#include "binning.hpp"
void identical_particles::quick_sort_init() {
    before = t_bool("before", N);
    after = t_bool("afater", N);
   
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
        KOKKOS_INLINE_FUNCTION   // Copy Constructor
            array_type(const int& rhs) {
            for (int i = 0; i < N; i++) {
                the_array[i] = rhs;
            }
        }
        KOKKOS_INLINE_FUNCTION   // add operator
            array_type& operator += (const array_type& src) {
            for (int i = 0; i < N; i++) {
                the_array[i] += src.the_array[i];
            }
            return *this;
        }
        KOKKOS_INLINE_FUNCTION   // constructor to convert from "int"
            array_type& operator = (int&& src) {
            for (int i = 0; i < N; i++) {
                the_array[i] = src;
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
/* This routine does the quicksort algorithm, it parallelize alny the partitioning  */
void identical_particles::create_quick_sort_v1() {
    
    quickSort(0, N - 1);
    

    Kokkos::parallel_for("quicksort-binncount", Kokkos::TeamPolicy<functor_count_bin::Tag_count_bin>(bintot, Kokkos::AUTO),
        functor_count_bin(N, nbin, sizebin, x, permute_vector_temp, permute_vector, bincount, binoffsets));

    Kokkos::parallel_scan("quicksort-binoffsets", Kokkos::RangePolicy<functor_count_bin::Tag_offset>(0, bintot),
        functor_count_bin(N, nbin, sizebin, x, permute_vector_temp, permute_vector, bincount, binoffsets));

    Kokkos::parallel_for("binset", Kokkos::TeamPolicy<functor_count_bin::Tag_set_bin>(bintot, Kokkos::AUTO),
        functor_count_bin(N, nbin, sizebin, x, permute_vector_temp, permute_vector, bincount, binoffsets));

    Kokkos::parallel_for("copy-range", Kokkos::RangePolicy(0, N), copy_range(permute_vector, permute_vector_temp));
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////




KOKKOS_INLINE_FUNCTION int ctolex(int bx, int by, int bz, int nbin[dim_space]) {
    return bx + nbin[0] * (by + bz * nbin[1]);
};

KOKKOS_INLINE_FUNCTION int which_bin(int i, double sizebin[dim_space], int nbin[dim_space], type_x x) {
    const int bx = floor(x(i, 0) / sizebin[0]);
    const int by = floor(x(i, 1) / sizebin[1]);
    const int bz = floor(x(i, 2) / sizebin[2]);
    // printf("i=%d --> %d %d %d  ---> %d\n", i, bx, by, bz, ctolex(bx, by, bz));
    return ctolex(bx, by, bz, nbin);
};
/* This function takes last element as pivot, places
 *  the pivot element at its correct position in sorted
 *   array, and places all smaller (smaller than pivot)
 *  to left of pivot and all greater elements to right
 *  of pivot */
template <class TeamMember>
KOKKOS_INLINE_FUNCTION int partition_middleTask(TeamMember& member, int nbin[dim_space], double sizebin[dim_space],
    type_x& x, t_permute_vector& permute_vector,
    t_permute_vector& permute_vector_temp, t_bool& before, t_bool& after, int low, int high) {
    int pi = (high + low) / 2 + 1;
    int pi_pos = 0;
    Kokkos::parallel_reduce(//"compare_to_pivot",
        // Kokkos::RangePolicy<quicksort_compare::Tag_no_pivot>(low, high + 1),
            // quicksort_compare(pi, nbin, sizebin, x, permute_vector, before, after), pi_pos);
        Kokkos::TeamThreadRange(member, low, high + 1),
        [=](const int j, int& update) {
            if (j != pi) {
                const int a = which_bin(permute_vector(j), sizebin, nbin, x);
                const int b = which_bin(permute_vector(pi), sizebin, nbin, x);
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

        }, pi_pos);
    pi_pos += low;
    Kokkos::parallel_scan(//"prefix_and_set",
        // Kokkos::RangePolicy(low, high + 1), functor_prefix_and_set(permute_vector_temp, permute_vector, before, after, low, pi_pos)
        Kokkos::TeamThreadRange(member, low, high + 1),
        [=](const int j, arrayN2::ValueType& update, const bool final) {
            // Load old value in case we update it before accumulating
            int b_j = before(j);
            int a_j = after(j);

            if (before(j) == 1) {
                if (final) {
                    b_j = update.the_array[0];// only update array on final pass
                    permute_vector_temp(b_j + low) = permute_vector(j);
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
        }
    );
    Kokkos::parallel_for(//"set-pivot",
        //  Kokkos::RangePolicy(pi_pos, pi_pos + 1), functor_set_pivot(permute_vector_temp, permute_vector, pi)
        Kokkos::TeamThreadRange(member, pi_pos, pi_pos + 1),
        [=](const int will_be) {
            permute_vector_temp(will_be) = permute_vector(pi);
        }
    );

    Kokkos::parallel_for(//"copy-range",
        //  Kokkos::RangePolicy(low, high + 1), copy_range(permute_vector, permute_vector_temp)
        Kokkos::TeamThreadRange(member, low, high + 1),
        [=](const int j) {
            permute_vector(j) = permute_vector_temp(j);}
    );
    return pi_pos;
}


template <class Scheduler>
struct quickSortTask {
    using value_type = int;
    using future_type = Kokkos::BasicFuture<int, Scheduler>;
    int nbin[dim_space];
    double sizebin[dim_space];
    int low;
    int high;
    type_x x;
    t_permute_vector permute_vector;
    t_permute_vector permute_vector_temp;
    t_bool before;
    t_bool after;
    future_type fn_1;
    future_type fn_2;


    KOKKOS_INLINE_FUNCTION
        explicit
        quickSortTask(int nbin_[dim_space], double sizebin_[dim_space], type_x& x_, t_permute_vector& permute_vector_,
            t_permute_vector& permute_vector_temp_, t_bool& before_, t_bool& after_, int low_, int high_) noexcept
        :
        nbin{ nbin_[0], nbin_[1], nbin_[2] },
        sizebin{ sizebin_[0], sizebin_[1], sizebin_[2] },
        low(low_),
        high(high_),
        x(x_),
        permute_vector(permute_vector_),
        permute_vector_temp(permute_vector_temp_),
        before(before_),
        after(after_) {
    }

    template <class TeamMember>
    KOKKOS_INLINE_FUNCTION void operator()(TeamMember& member, int& result) {
        auto& scheduler = member.scheduler();
        if (low < high) {

            int pi = partition_middleTask(member, nbin, sizebin, x, permute_vector,
                permute_vector_temp, before, after, low, high);
            // int pi = (low + high) / 2;

            // Separately sort elements before
            // partition and after partition
            // quickSort(low, pi - 1);
            Kokkos::single(Kokkos::PerTeam(member), [&]() {
                fn_1 = Kokkos::task_spawn(
                    Kokkos::TaskTeam(scheduler),
                    quickSortTask<Scheduler>(nbin, sizebin, x, permute_vector,
                        permute_vector_temp, before, after, low, pi - 1));
                });
            // quickSort(pi + 1, high);
            Kokkos::single(Kokkos::PerTeam(member), [&]() {
                fn_2 = Kokkos::task_spawn(
                    Kokkos::TaskTeam(scheduler),
                    quickSortTask(nbin, sizebin, x, permute_vector,
                        permute_vector_temp, before, after, pi + 1, high));
                });
        }

        // Create an aggregate predecessor for our respawn // not sure if we need to wait here, maybe it is done authomatically
        Kokkos::BasicFuture<void, Scheduler> quickSort_array[] = { fn_1, fn_2 };
        auto f_all = scheduler.when_all(quickSort_array, 2);

        // Respawn this task with `f_all` as a predecessor
        // Kokkos::respawn(this, f_all);
    }
};




size_t estimate_required_memory(int N) {
    assert(N >= 0);
    auto nl = static_cast<size_t>(N);
    return (nl + 1) * 2000 * sizeof(int);// no idea this number how much should be
}



/* This routine does the full parallelised quicksort algorithm */
void identical_particles::create_quick_sort() {
    Kokkos::Timer t1;
    // quickSort(0, N - 1);
    /////////////////////////////
    using scheduler_type = Kokkos::TaskScheduler<Kokkos::DefaultExecutionSpace>;
    using memory_space = typename scheduler_type::memory_space;
    using memory_pool = typename scheduler_type::memory_pool;

    auto mpool = memory_pool(memory_space{}, estimate_required_memory(N));
    auto scheduler = scheduler_type(mpool);

    Kokkos::BasicFuture<int, scheduler_type> result;

    Kokkos::Timer timer;
    {
        // launch the root task from the host
        result =
            Kokkos::host_spawn(
                Kokkos::TaskTeam(scheduler),
                quickSortTask<scheduler_type>(nbin, sizebin, x, permute_vector,
                    permute_vector_temp, before, after, 0, N - 1)
            );

        // wait on all tasks submitted to the scheduler to be done
        Kokkos::wait(scheduler);
    }

    Kokkos::parallel_for("quicksort-binncount", Kokkos::TeamPolicy<functor_count_bin::Tag_count_bin>(bintot, Kokkos::AUTO),
        functor_count_bin(N, nbin, sizebin, x, permute_vector_temp, permute_vector, bincount, binoffsets));

    Kokkos::parallel_scan("quicksort-binoffsets", Kokkos::RangePolicy<functor_count_bin::Tag_offset>(0, bintot),
        functor_count_bin(N, nbin, sizebin, x, permute_vector_temp, permute_vector, bincount, binoffsets));

    Kokkos::parallel_for("binset", Kokkos::TeamPolicy<functor_count_bin::Tag_set_bin>(bintot, Kokkos::AUTO),
        functor_count_bin(N, nbin, sizebin, x, permute_vector_temp, permute_vector, bincount, binoffsets));

    Kokkos::parallel_for("copy-range", Kokkos::RangePolicy(0, N), copy_range(permute_vector, permute_vector_temp));
};


