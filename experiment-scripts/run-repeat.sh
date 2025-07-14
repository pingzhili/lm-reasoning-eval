#!/bin/bash

# Function to run each script 3 times sequentially
run_script_three_times() {
    local script=$1
    local script_basename=$(basename "$script" .sh)

    for i in {1..4}; do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting $script_basename - Run $i/4"
        ./"$script"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed $script_basename - Run $i/4"
        echo "----------------------------------------"
    done

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] All runs completed for $script_basename"
    echo "========================================"
}

# Start time
echo "Starting parallel execution of qwen3-32b scripts"
echo "Each script will run 4 times sequentially"
echo "========================================"
start_time=$(date +%s)

# Run each script's 4 iterations in parallel
run_script_three_times "experiment-scripts/qwen3-32b-non-thinking.sh" &
pid1=$!

run_script_three_times "experiment-scripts/qwen3-32b-self-judge.sh" &
pid2=$!

run_script_three_times "experiment-scripts/qwen3-32b-thinking.sh" &
pid3=$!

# Wait for all background jobs to complete
echo "Waiting for all scripts to complete..."
wait $pid1 $pid2 $pid3

# End time and duration
end_time=$(date +%s)
duration=$((end_time - start_time))

echo "========================================"
echo "All scripts completed successfully!"
echo "Total execution time: $duration seconds"
echo "========================================"