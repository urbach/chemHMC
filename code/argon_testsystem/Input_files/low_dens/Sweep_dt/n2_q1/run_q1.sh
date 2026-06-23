#!/bin/bash

# Check for correct usage
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <dt> <ntraj>"
    exit 1
fi

DT="$1"
NTRAJ="$2"

DIR="/home/marko/Faks/PhD/Projects/Symplectic_integrators/chemHMC/code"
SAVE_PATH="${DIR}/argon_testsystem/Data/low_dens/Sweep_dt/n2_q1/argon_low_${DT}.xyz"
cp "${DIR}/argon_testsystem/Input_files/low_dens/argon.xyz" "$SAVE_PATH"

INPUT_FILE="${DIR}/argon_testsystem/Input_files/low_dens/Sweep_dt/n2_q1/argon_low_${DT}.yaml"

# Generate the input file
cat << EOF > "$INPUT_FILE"
parameter_file: ${DIR}/argon_testsystem/Input_files/low_dens/parameters.inp
start_configuration_file: ${DIR}/argon_testsystem/Input_files/low_dens/argon.xyz
output_file: ${SAVE_PATH}
seed: 123
Ntrajectories: ${NTRAJ}
thermalization_steps: 0
save_every: 1
print_info_every: 1000
simulation_type: MD

geometry: 
  Lx: 100.0
  Ly: 100.0
  Lz: 100.0

particles:
  MaxNeighbors: 450
  temperature: 300.0

LJ:
  cutoff: 12.0
  algorithm: verlet_list
  shift_potential:

integrator:
  name: GENERAL
  dt: ${DT}
  steps: 1
  cycles: 1
  a: [0.5, 0.5]
  b: [1.0]

EOF

# Run your program with the generated input
${DIR}/builds/openMP/main/main -i "$INPUT_FILE"

rm "$INPUT_FILE"
