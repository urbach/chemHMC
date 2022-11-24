#!/bin/bash
# this line is a comment
# remove chache
rm -r CMakeFiles CMakeCache.txt

cd ..
projectHOME=`pwd`
cd build
echo "project HOME:" $projectHOME

CXXFLAGS=" " \
cmake .. \
  -DCMAKE_BUILD_TYPE=DEBUG \
  -DKokkos_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_CXX_STANDARD=17 \
  -DKokkos_ENABLE_SERIAL=ON  \
  -DKokkos_ARCH_KEPLER35=ON \
  -DKokkos_ENABLE_CUDA_LAMBDA=ON \
  -DCMAKE_PREFIX_PATH="/hiskp4/garofalo/yaml-cpp/install_dir" \
  -DCMAKE_CXX_COMPILER=${projectHOME}/external/kokkos/bin/nvcc_wrapper 

