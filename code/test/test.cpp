#include <Kokkos_Core.hpp>

class a_class {
public:
    std::string sa;
    KOKKOS_FUNCTION void operator() (const int& i, double& s1) const { s1++; };
};


int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        a_class myclass;
        double sum;
        Kokkos::parallel_reduce("myclass", 1, myclass, sum);

    }
    Kokkos::finalize();
}
