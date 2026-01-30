#!/bin/bash
# Launch script for data size sweep experiments
# Runs nano and tiny models with 500, 1000, 5000, and all data
#
# Usage:
#   ./launch_data_size_sweep.sh [CLUSTER] [PRIORITY] [NUM_GPUS]
#
# Example:
#   ./launch_data_size_sweep.sh ai2/ceres-cirrascale urgent 4

set -e  # Exit on error

# Default values (can be overridden via command line)
CLUSTER="${1:-ai2/ceres-cirrascale}"
PRIORITY="${2:-urgent}"
NUM_GPUS="${3:-4}"

# Model sizes and data sizes to run
MODELS=("nano" "tiny")
DATA_SIZES=(500 1000 5000 "all")

# Script paths (relative to project root)
SCRIPT_DIR="scripts/official"
NANO_SCRIPT="$SCRIPT_DIR/nano.py"
TINY_SCRIPT="$SCRIPT_DIR/tiny.py"

# Total dataset size (from h5py_dir name in script.py)
TOTAL_DATASET_SIZE=1138828

echo "=========================================="
echo "Launching data size sweep experiments"
echo "=========================================="
echo "Cluster: $CLUSTER"
echo "Priority: $PRIORITY"
echo "Num GPUs: $NUM_GPUS"
echo "Models: ${MODELS[*]}"
echo "Data sizes: ${DATA_SIZES[*]}"
echo "=========================================="
echo ""

# Function to calculate dataset percentage
calculate_percentage() {
    local num_samples=$1
    
    if [ "$num_samples" == "all" ]; then
        echo "1.0"
    else
        # Use awk for floating point calculation (more portable than bc)
        awk "BEGIN {printf \"%.10f\", $num_samples / $TOTAL_DATASET_SIZE}"
    fi
}

# Launch all combinations
for model in "${MODELS[@]}"; do
    # Select the appropriate script
    if [ "$model" == "nano" ]; then
        SCRIPT="$NANO_SCRIPT"
    elif [ "$model" == "tiny" ]; then
        SCRIPT="$TINY_SCRIPT"
    else
        echo "Error: Unknown model '$model'"
        exit 1
    fi
    
    for data_size in "${DATA_SIZES[@]}"; do
        # Create run name
        if [ "$data_size" == "all" ]; then
            run_name="${model}_data_all"
        else
            run_name="${model}_data_${data_size}"
        fi
        
        # Calculate dataset percentage
        dataset_percentage=$(calculate_percentage "$data_size")
        
        echo "Launching: $run_name"
        echo "  Model: $model"
        echo "  Data size: $data_size"
        echo "  Dataset percentage: $dataset_percentage"
        
        # Launch the experiment
        python "$SCRIPT" launch "$run_name" "$CLUSTER" \
            --dataset.dataset_percentage="$dataset_percentage" \
            --launch.clusters="[$CLUSTER]" \
            --launch.priority="$PRIORITY" \
            --launch.num_gpus="$NUM_GPUS"
        
        echo ""
    done
done

echo "=========================================="
echo "All experiments launched!"
echo "=========================================="

