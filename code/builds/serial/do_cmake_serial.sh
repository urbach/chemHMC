#!/bin/bash
# remove chache
rm -r CMakeFiles CMakeCache.txt cmake_install.cmake

CXXFLAGS="-O2" \
cmake ../../ \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DKokkos_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_CUDA=OFF \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ARCH_NATIVE=ON \
  -DCMAKE_PREFIX_PATH="/usr/include/yaml-cpp/"   
