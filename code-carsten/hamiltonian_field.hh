#pragma once

#include"config.hh"
#include<vector>

template<class T> struct hamiltonian_field {
  config<T> * momenta;
  config<T> * U;
  hamiltonian_field(config<T> &momenta, config<T> &U) :
    momenta(&momenta), U(&U) {}
};
