# chemHMC
 
At this stage is this directory is just a template for a project with kokkos

# Download and Installation

In order to fetch the code:
```
    git clone git@github.com:urbach/chemHMC.git
    cd chemHMC
    cd code
    git submodule update --init --recursive
```

# Building

Kokkos can be build for different devices

## openMP build

you can use the script `do_cmake_openMP.sh`

``` 
   cd build
   bash do_cmake_openMP.sh
   make
```

## CUDA build

you can use the script `do_cmake_cuda.sh` , you may need to set up the architecture,

```
   -DKokkos_ARCH_KEPLER35=ON
```
a list of option can be found in https://github.com/kokkos/kokkos/wiki/Compiling. Then build and compile with

``` 
   cd build
   bash do_cmake_cuda.sh
   make
```

# Test and tutorial

so far the only executaprogramble present is
```
   test/kokkos_tutorial4.cpp
```
that after building and compilation there will be the executable  
```
   build/test/kokkos_tutorial4
```