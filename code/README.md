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

# Download 

To fetch the code:
```
git clone git@github.com:urbach/chemHMC.git
cd chemHMC
cd code
git submodule update --init --recursive
```

# Building

Kokkos can be built for different devices

## openMP build

you can use the script `do_cmake_openMP.sh`

``` 
cd builds/openMP
bash do_cmake_openMP.sh
make
```

## CUDA build

you can check the directories
```
builds/qbig_ampere
builds/qbig_kepler
builds/qbig_pascal
```
for some builds with CUDA, a list of option can be found in https://kokkos.github.io/kokkos-core-wiki/keywords.html.

Make sure the `default_arch` in `code/external/kokkos/bin/nvcc_wrapper` matches the lowest supported version shown in `nvcc --list-gpu-arch`.
By default `sm_35` is set there which is suitable for the above specified Kepler architecture.


# Infile 

the infile is written in the yaml format, an example is in 
```
/build/test.yaml
```

## Lennard-Jones cutoff

How the LJ interaction is brought to zero at `cutoff` is set with `cutoff_scheme` in the `LJ` block
(works for both `algorithm: verlet_list` and `algorithm: cutoff`):

| `cutoff_scheme` | potential | at the cutoff |
|---|---|---|
| `none` (default) | plain truncation | energy and force jump |
| `shift` | V(r) - V(r_c) | energy continuous, force jumps (LAMMPS `pair_modify shift`, GROMACS `Potential-shift`) |
| `force_shift` | V(r) - V(r_c) - (r - r_c) V'(r_c) | energy and force continuous, V'' jumps |
| `switch` | V(r) S(r), C^4 polynomial switch from `switch_on` to `cutoff` | V and its first four derivatives continuous |

```
LJ:
  cutoff: 12.0
  algorithm: verlet_list
  cutoff_scheme: switch
  switch_on: 10.0
```
The older key `shift_potential` is still accepted (`shift_potential:` or `shift_potential: true` means `shift`,
`shift_potential: false` means `none`).

A jump in the k-th derivative of V at the cutoff limits the energy error of every integrator to O(dt^k)
once pairs cross the cutoff. To measure the order of a high-order integrator use `force_shift` or `switch`;
`shift` is the usual production choice and its force jump is only visible at very small energy errors.

## General splitting integrators

`GENERAL_AB` and `GENERAL_BA` run any splitting scheme for H = A + B, with A the kinetic part
(drift, position update, coefficients `a`) and B the potential part (kick, momentum update, coefficients `b`).
With `cycles` = q, a step costs q force evaluations.

| `name` | sequence | `a` (drifts) | `b` (kicks) |
|---|---|---|---|
| `GENERAL_AB` | drift first: Q(a0) P(b0) ... P(b_{q-1}) Q(a_q) | q+1 | q |
| `GENERAL_BA` | kick first: P(b0) Q(a0) ... Q(a_{q-1}) P(b_q) | q | q+1 |

This matches the scheme files in `argon_testsystem/Data/Schemes/<precision>/order_n*/cycles_q*/{AB,BA}`:
`standard_parameters_a` and `standard_parameters_b` hold the first halves of the symmetric `a` and `b`.
Velocity Verlet is
```
integrator:
  name: GENERAL_BA
  dt: 1.0
  steps: 1
  cycles: 1
  a: [1.0]
  b: [0.5, 0.5]
```
The former `GENERAL` (kick first, `a` = kicks) is `GENERAL_BA` with `a` and `b` swapped.

# Test and tutorial

The executable 
```
build/main/main
```
will be generated in the compilation, assuming you are in the directory `build/main/`, it can be executed as
```
./main -i ../test.yaml
```
