#!/bin/bash

# Check input arguments
if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 <path_to_dt_values.in> <T> <path_to_run_script.sh> [cutoff_scheme]"
    echo "  T              total simulated time in fs, the same for every dt"
    echo "  cutoff_scheme  shift (default, production) | force_shift | switch | none"
    exit 1
fi

DT_FILE="$1"
T="$2"
RUN_SCRIPT="$3"
CUT="${4:-shift}"

# Validate file paths
if [ ! -f "$DT_FILE" ]; then
    echo "Error: dt-values file '$DT_FILE' not found!"
    exit 2
fi

if [ ! -x "$RUN_SCRIPT" ]; then
    echo "Error: run script '$RUN_SCRIPT' not found or not executable!"
    exit 3
fi

# Loop through dt values (also reads a last line without a newline, skips blank lines)
while IFS= read -r dtval || [ -n "$dtval" ]; do
    dtval="${dtval//[[:space:]]/}"
    [ -z "$dtval" ] && continue
    echo "Running for dt = $dtval"
    "$RUN_SCRIPT" "$dtval" "$T" "$CUT" < /dev/null

    if [ $? -ne 0 ]; then
        echo "Error: Script failed for dt = $dtval"
        exit 4
    fi
done < "$DT_FILE"
