#ifndef GLOBAL_H
#define GLOBAL_H

#ifdef CONTROL 
#define EXTERN 
#else
#define EXTERN extern
#endif

// Useful constants
static const double N_A = 6.02214076e23;
static const double kcaltointernal = 4.184e-4; //conversion from kcal/mol to amu * A^2/fs^2
static const double kjtointernal = 1.0e-4; //conversion from kJ/mol to amu * A^2/fs^2
static const double kB_J = 1.380649e-23; //boltzmann constant in J/K
static const double kB = kB_J * 1e-3 * N_A * kjtointernal; //boltzmann constant in amu * A^2/(fs^2*K)

#define dim_space 3

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
typedef typename Kokkos::Random_XorShift64_Pool<> RandPoolType;
typedef typename RandPoolType::generator_type gen_type;
typedef Kokkos::View<double* [dim_space]> type_x;
typedef Kokkos::View<double* [dim_space]> type_p;
typedef Kokkos::View<double* [dim_space]> type_f;
typedef Kokkos::View<int*> type_id;
typedef Kokkos::View<const double* [dim_space]> type_const_x;
typedef Kokkos::View<const double* [dim_space]> type_const_p;
typedef Kokkos::View<const double* [dim_space]> type_const_f;
typedef Kokkos::View<const int*> type_const_id;
typedef Kokkos::TeamPolicy<>::member_type  member_type;


#endif
