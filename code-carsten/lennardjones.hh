#pragma once

#include<cmath>

inline double LennardJones(const double r, const int n=12,
                           const double a=1.e-12, const double b=1.e-6) {
  
  return(a/pow(r, n) - b/pow(r, 6));
}
