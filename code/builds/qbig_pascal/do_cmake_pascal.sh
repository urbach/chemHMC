#!/bin/bash
# this line is a comment
# remove chache
#set -x
rm -r CMakeFiles CMakeCache.txt

cd ../..
projectHOME=`pwd`
cd -
echo "project HOME:" $projectHOME

source load_modules_qbig_pascal.sh

CXXFLAGS="-mtune=sandybridge -march=sandybridge -g -O3" \
cmake ../.. \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DKokkos_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ENABLE_CUDA_LAMBDA=ON \
  -DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON \
  -DKokkos_ENABLE_DEBUG=OFF \
  -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=OFF \
  -DKokkos_ENABLE_COMPILER_WARNINGS=ON \
  -DCMAKE_CXX_STANDARD=17 \
  -DKokkos_ARCH_KEPLER35=OFF \
  -DKokkos_ARCH_PASCAL60=ON \
  -DKokkos_ENABLE_TESTS=OFF \
  -DCMAKE_PREFIX_PATH="/hiskp4/garofalo/yaml-cpp/install_dir" \
  -DCMAKE_CXX_COMPILER=${projectHOME}/external/kokkos/bin/nvcc_wrapper 

  #-DKokkos_ENABLE_SERIAL=ON  \
# -DKOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON \ # is necessary to use tasks
