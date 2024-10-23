#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include <Kokkos_Core.hpp>



KOKKOS_FUNCTION void times_2(int& i) {
    i = i * 2;
}

class myclass_kernel {
public:
    struct tag_for {};
    struct tag_reduce {};

    KOKKOS_FUNCTION void operator() (tag_for, const int i) const;
    KOKKOS_FUNCTION void operator() (tag_reduce, const int i, double &sum) const;
    void call_for();
    void call_reduce();
};


class myclass {
public:
    struct tag_for {};
    struct tag_reduce {};
    std::string myname = "lalalal";
    myclass_kernel kernels;
    // KOKKOS_FUNCTION void operator() (tag_for, const int i) const;
    // KOKKOS_FUNCTION void operator() (tag_reduce, const int i, double &sum) const;
    void call_for();
    void call_reduce();
};

void myclass::call_for(){
    kernels.call_for();
};
void myclass::call_reduce(){
    kernels.call_reduce();
};


KOKKOS_FUNCTION
void myclass_kernel::operator() (tag_for, const int i) const {
    printf("do nothing %d\n", i);
    int a = i;
    times_2(a);
    printf("a= %d\n", a);
};
void myclass_kernel::call_for() {
    Kokkos::parallel_for("calling for", Kokkos::RangePolicy<tag_for>(0, 10), *this);
};
KOKKOS_FUNCTION
void myclass_kernel::operator() (tag_reduce, const int i, double& sum) const {
    sum++;
    printf("do sum %d\n", i);
    int a = i;
    times_2(a);
    printf("a= %d\n", a);

};
void myclass_kernel::call_reduce() {
    double a = 0;
    Kokkos::parallel_reduce("calling for", Kokkos::RangePolicy<tag_reduce>(0, 10), *this, a);
    printf("reduce result= %g\n", a);
};


int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);

    {
       

        printf("functor test");
        myclass a;
        a.call_for();
        a.call_reduce();


    }

    Kokkos::finalize();
}
