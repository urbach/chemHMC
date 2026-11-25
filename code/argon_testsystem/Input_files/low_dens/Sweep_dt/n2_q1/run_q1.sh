#!/bin/bash
# Velocity Verlet / leapfrog (GENERAL_BA: kick-drift-kick, a = drifts, b = kicks) on the low-density argon box.
# Every run integrates for the same physical time T, so runs with different dt are comparable.

# Check for correct usage
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <dt> <T> [cutoff_scheme]"
    echo "  dt, T          time step and total simulated time in fs (T must be a multiple of dt)"
    echo "  cutoff_scheme  shift (default, production) | force_shift | switch | none"
    exit 1
fi

DT="$1"
T="$2"
CUT="${3:-shift}"

# Number of steps needed to reach T
NTRAJ=$(python3 -c "n = round($T / $DT); assert n >= 1 and abs(n * $DT - $T) < 1e-9 * $T; print(n)" 2>/dev/null) || {
    echo "Error: T = $T fs must be a positive multiple of dt = $DT fs"
    exit 1
}

case "$CUT" in
    shift)       LJ_CUTOFF="  cutoff_scheme: shift" ;;
    force_shift) LJ_CUTOFF="  cutoff_scheme: force_shift" ;;
    switch)      LJ_CUTOFF=$'  cutoff_scheme: switch\n  switch_on: 10.0' ;;
    none)        LJ_CUTOFF="  cutoff_scheme: none" ;;
    *)           echo "Error: unknown cutoff_scheme '$CUT'"; exit 1 ;;
esac

echo "dt = $DT fs, T = $T fs: $NTRAJ steps, cutoff_scheme = $CUT"

DIR="/home/marko/Faks/PhD/Projects/Symplectic_integrators/chemHMC/code"
DATA_DIR="${DIR}/argon_testsystem/Data/low_dens/Sweep_dt/n2_q1/T${T}_${CUT}"
mkdir -p "$DATA_DIR"
SAVE_PATH="${DATA_DIR}/argon_low_${DT}.xyz"
# The program appends to the output file, so remove the data of an earlier run
rm -f "$SAVE_PATH"

INPUT_FILE="${DIR}/argon_testsystem/Input_files/low_dens/Sweep_dt/n2_q1/argon_low_T${T}_${CUT}_${DT}.yaml"

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
${LJ_CUTOFF}

integrator:
  name: GENERAL_BA
  dt: ${DT}
  steps: 1
  cycles: 1
  a: [1.0]
  b: [0.5, 0.5]

EOF

# Run your program with the generated input
${DIR}/builds/openMP/main/main -i "$INPUT_FILE"
STATUS=$?

rm "$INPUT_FILE"
# Pass the program's exit status on, so sweep.sh stops when a run fails
exit $STATUS
