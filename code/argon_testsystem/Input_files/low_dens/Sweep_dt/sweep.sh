#!/bin/bash

# Check input arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <path_to_dt_values.in> <ntraj> <path_to_run_script.sh>"
    exit 1
fi

DT_FILE="$1"
NTRAJ="$2"
RUN_SCRIPT="$3"

# Validate file paths
if [ ! -f "$DT_FILE" ]; then
    echo "Error: dt-values file '$DT_FILE' not found!"
    exit 2
fi

if [ ! -x "$RUN_SCRIPT" ]; then
    echo "Error: run script '$RUN_SCRIPT' not found or not executable!"
    exit 3
fi

# Loop through dt values
while IFS= read -r dtval; do
    echo "Running for dt = $dtval"
    "$RUN_SCRIPT" "$dtval" "$NTRAJ"

    if [ $? -ne 0 ]; then
        echo "Error: Script failed for dt = $dtval"
        exit 4
    fi
done < "$DT_FILE"
