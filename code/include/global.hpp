#ifndef GLOBAL_H
#define GLOBAL_H

#ifdef CONTROL 
#define EXTERN 
#else
#define EXTERN extern
#endif


#define dim_space 3

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
typedef typename Kokkos::Random_XorShift64_Pool<> RandPoolType;
typedef typename RandPoolType::generator_type gen_type;

#endif
