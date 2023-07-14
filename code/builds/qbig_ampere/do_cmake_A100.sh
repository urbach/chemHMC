#!/bin/bash
# this line is a comment
# remove chache
#set -x
rm -r CMakeFiles CMakeCache.txt

cd ../..
projectHOME=`pwd`
cd -
echo "project HOME:" $projectHOME



CXXFLAGS=" " \
cmake ${projectHOME} \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DKokkos_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ENABLE_CUDA_LAMBDA=ON \
  -DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON \
  -DKokkos_ENABLE_DEBUG=OFF \
  -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=OFF \
  -DKokkos_ENABLE_COMPILER_WARNINGS=ON \
  -DCMAKE_CXX_STANDARD=17 \
  -DKokkos_ENABLE_SERIAL=ON  \
  -DKokkos_ARCH_AMPERE80=ON \
  -DKokkos_ENABLE_CUDA_LAMBDA=ON \
  -DKokkos_ENABLE_TESTS=OFF \
  -DCMAKE_PREFIX_PATH="/qbigwork2/garofalo/yaml-cpp/install_dir/" \
  -DCMAKE_CXX_FLAGS=" -L/qbigwork2/garofalo/yaml-cpp/install_dir/lib"
  -DCMAKE_CXX_COMPILER=${projectHOME}/external/kokkos/bin/nvcc_wrapper 

# -DKOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON \ # is necessary to use tasks
