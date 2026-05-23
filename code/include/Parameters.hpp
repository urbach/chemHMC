#ifndef PARAMETERS_H
#define PARAMETERS_H

#include "global.hpp"

struct params_class {
  double L[dim_space];
  double inverse_L[dim_space];
  double inverse_halved_L[dim_space];
  double pressure; // Input pressure in atmospheres
  double volume_step; // Max step size for volume moves
  int Ntrajectories;
  int thermalization_steps;
  int save_every;
  int print_info_every;
  // run parameter
  int seed;
  int N;

  bool hb_momenta = true;

  FILE* fileout;
  std::string start_configuration_file;
  std::string nameout;
  std::string parameter_file;
};

#endif