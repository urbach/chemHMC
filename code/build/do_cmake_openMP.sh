#!/bin/bash
# this line is a comment
# remove chache
rm -r CMakeFiles CMakeCache.txt cmake_install.cmake

CXXFLAGS="-O2 -fopenmp -g" \
cmake .. \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_CUDA=OFF \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_CXX_STANDARD=14  \
  -DCMAKE_PREFIX_PATH="/home/garofalo/programs/yaml-cpp/install_dir"  

 
