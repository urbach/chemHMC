# chemHMC
 
At this stage is this directory is just a template for a project with kokkos

# Prerequisites

we need 

* [yaml-cpp](https://github.com/jbeder/yaml-cpp)

to install it you can try
```
sudo apt-get install libyaml-cpp-dev
```
or install manually
```
git clone https://github.com/jbeder/yaml-cpp
cd yaml-cpp
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX="../install_dir" ..
make
make install
```

Then in the `do_cmake_$$$.sh` file you need to specify 
```
-DCMAKE_PREFIX_PATH="installation/dir/of/yaml-cpp"
```

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
a list of option can be found in https://kokkos.github.io/kokkos-core-wiki/keywords.html.

Make sure the `default_arch` in `code/external/kokkos/bin/nvcc_wrapper` matches the lowest supported version shown in `nvcc --list-gpu-arch`.
By default `sm_35` is set there which is suitable for the above specified Kepler architecture.

Then build and compile with

``` 
cd build
bash do_cmake_cuda.sh
make
```
# Infile 

the infile is written in the yaml format, an example is in 
```
/build/test.yaml
```

# Test and tutorial

The executable 
```
build/main/main
```
will be generated in the compilation, assuming you are in the directory `build/main/`, it can be executed as
```
./main -i ../test.yaml
```
