#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --qos=devel
# #SBATCH --partition=devel
#SBATCH --partition=batch
#SBATCH --reservation=debug
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:pascal:1        # 2 gpus per node out of 8 ?
#SBATCH --mem=15GB

source load_modules_qbig_pascal.sh
#export KOKKOS_PROFILE_LIBRARY=/hiskp4/garofalo/chemHMC/code/external/kokkos-tools/kp_memory_events.so
# export KOKKOS_PROFILE_LIBRARY=/hiskp4/garofalo/chemHMC/code/external/kokkos-tools/kp_kernel_logger.so

 
#rm rng* out_xyz.txt
#/qbigwork/garofalo/valgrind/install_dir/bin/valgrind --leak-check=full ../../chemHMC/code/build/main//main -i input_I.yaml
#../../chemHMC/code/build/main//main -i input_I.yaml
#main/main -i ../test.yaml
#./test/test 
#./test/test_binning -i ../test.yaml                                         
