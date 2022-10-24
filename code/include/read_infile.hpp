#ifndef READ_INFILE_H
#define READ_INFILE_H

#include "global.hpp"
#include "yaml-cpp/yaml.h"

struct params_class { // Just the thing that holds all variables
  
  double L[dim_space];
  std::string StartCondition;
  
  // run parameter
  int seed;
  int replica;
  int start_measure;
  int total_measure;
  int measure_every_X_updates;
  // do nothing in the constructor
  params_class(){};
  // using this function to initialize
  YAML::Node read_params(int argc, char** argv );// function that reads the infile

};


template<class T> T check_and_assign_value(YAML::Node doc, const char *tag);

#endif