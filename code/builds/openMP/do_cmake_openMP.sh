#!/bin/bash
# remove chache
rm -r CMakeFiles CMakeCache.txt cmake_install.cmake

CXXFLAGS="-O2 -fopenmp -g" \
cmake ../../ \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_CUDA=OFF \
  -DKokkos_ENABLE_SERIAL=ON \
  -DCMAKE_PREFIX_PATH="/home/marko/Faks/PhD/Projects/Symplectic_integrators/chemHMC/code/external/yaml-cpp/install_dir/"
