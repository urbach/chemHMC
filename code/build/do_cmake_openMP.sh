#!/bin/bash
# this line is a comment
# remove chache
rm -r CMakeFiles CMakeCache.txt

CXXFLAGS="-O0 " \
cmake .. \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_BUILD_TYPE=DEBUG \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_CUDA=OFF \
  -DKokkos_ENABLE_SERIAL=ON

