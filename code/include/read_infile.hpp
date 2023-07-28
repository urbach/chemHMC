#ifndef READ_INFILE_H
#define READ_INFILE_H

#include "global.hpp"
#include "yaml-cpp/yaml.h"

struct params_class { // Just the thing that holds all variables
  
  double L[dim_space];
  // run parameter
  bool append;
  int seed;
  int replica;
  int start_measure;
  int total_measure;
  int measure_every_X_updates;
  int istart;

  FILE* fileout;
  std::string StartCondition;
  std::string start_configuration_file;
  std::string nameout;
  std::string rng_host_state;
  std::string rng_device_state;

  std::string name_RDF;
  
  // contructor;
  params_class(YAML::Node doc);
 

};


template<class T> T check_and_assign_value(YAML::Node doc, const char *tag);
YAML::Node read_params(int argc, char** argv);

#endif