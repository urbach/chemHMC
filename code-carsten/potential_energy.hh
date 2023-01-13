#pragma once

#include"config.hh"
#include"lennardjones.hh"

template<class T> double potential_energy(config<T> & U) {
  double res = 0;

  for(size_t i = 0; i < U.getN()-1; i++) {
    for(size_t j = i+1; j < U.getN(); j++) {
      res += LennardJones((U[i]-U[j]).norm());
    }
  }
  return(res);
}
