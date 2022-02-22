#ifndef READ_INFILE_H
#define READ_INFILE_H

#include "global.hpp"

struct DataContainer { // Just the thing that holds all variables
  
  double L[dim_space];
  int Nathoms;
  
  
  // run parameter
  int seed;
  int replica;
  int start_measure;
  int total_measure;
  int measure_every_X_updates;


  DataContainer(int argc, char** argv );// constructor declaration

};

#endif