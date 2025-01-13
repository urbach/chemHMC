#ifndef READ_INFILE_H
#define READ_INFILE_H

#include "global.hpp"
#include "yaml-cpp/yaml.h"

struct params_class {
  double L[dim_space];
  double inverse_L[dim_space];
  double inverse_halved_L[dim_space];
  int Ntrajectories;
  int thermalization_steps;
  int save_every;
  int print_info_every;
  // run parameter
  int seed;
  int N;

  FILE* fileout;
  std::string start_configuration_file;
  std::string nameout;
  std::string parameter_file;
};


template<class T> T check_and_assign_value(YAML::Node doc, const char *tag);

#endif