#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include <Kokkos_Core.hpp>



class Foo {
protected:
    int val;
public:


    KOKKOS_FUNCTION
        Foo();

    KOKKOS_FUNCTION
        virtual int value() { return 0; };

    KOKKOS_FUNCTION
        virtual ~Foo() {}
};

class Foo_1 : public Foo {
public:
    KOKKOS_FUNCTION
        Foo_1();

    KOKKOS_FUNCTION
        int value();
};

class Foo_2 : public Foo {
public:
    KOKKOS_FUNCTION
        Foo_2();

    KOKKOS_FUNCTION
        int value();
};


KOKKOS_FUNCTION
Foo::Foo() {
    val = 0;
}

KOKKOS_FUNCTION
Foo_1::Foo_1() : Foo() {
    val = 1;
}

KOKKOS_FUNCTION
int Foo_1::value() {
    return val;
}

KOKKOS_FUNCTION
Foo_2::Foo_2() : Foo() {
    val = 2;
}

KOKKOS_FUNCTION
int Foo_2::value() {
    return val;
}



int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);

    {
        Foo* f_1 = (Foo*)Kokkos::kokkos_malloc(sizeof(Foo_1));
        Foo* f_2 = (Foo*)Kokkos::kokkos_malloc(sizeof(Foo_2));

        Kokkos::parallel_for("CreateObjects", 1, KOKKOS_LAMBDA(const int&) {
            new ((Foo_1*)f_1) Foo_1();
            new ((Foo_2*)f_2) Foo_2();
        });

        int value_1, value_2;
        Kokkos::parallel_reduce("CheckValues", 1, KOKKOS_LAMBDA(const int&, int& lsum) {
            lsum = f_1->value();
        }, value_1);

        Kokkos::parallel_reduce("CheckValues", 1, KOKKOS_LAMBDA(const int&, int& lsum) {
            lsum = f_2->value();
        }, value_2);

        printf("Values: %i %i\n", value_1, value_2);
        Foo* ff_2 = new Foo_2();
        // Foo* ff_2 = (Foo*)Kokkos::kokkos_malloc(sizeof(Foo_2));
        value_1 = ff_2->value();
        // value_2 = ff_1->value();
        printf("Values: %i %i\n", value_1, value_2);

        Kokkos::parallel_for("DestroyObjects", 1, KOKKOS_LAMBDA(const int&) {
            f_1->~Foo();
            f_2->~Foo();
        });

        Kokkos::kokkos_free(f_1);
        Kokkos::kokkos_free(f_2);

    }

    Kokkos::finalize();
}
