#ifndef PARAMETERS_H
#define PARAMETERS_H

#include "global.hpp"

struct Parameters { 
  double L[dim_space];
  double inverse_L[dim_space];
  double inverse_halved_L[dim_space];
  // run parameter
  bool append;
  int seed;
  int istart;
  int N;

  FILE* fileout;
  std::string StartCondition;
  std::string start_configuration_file;
  std::string nameout;
  std::string parameter_file;
};

#endif