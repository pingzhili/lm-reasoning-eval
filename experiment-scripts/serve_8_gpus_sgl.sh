#!/bin/bash

# Start python3 -m sglang.launch_server --model on 8 GPUs with ports 8000-8007
MODEL_NAME="openai/gpt-oss-120b"

echo "Starting python3 -m sglang.launch_server --model on 8 GPUs..."

# Start servers on each GPU
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server --model $MODEL_NAME --log-requests --log-requests-level 3 --port 8000 &
CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server --model $MODEL_NAME --log-requests --log-requests-level 3 --port 8001 &
CUDA_VISIBLE_DEVICES=2 python3 -m sglang.launch_server --model $MODEL_NAME --log-requests --log-requests-level 3 --port 8002 &
CUDA_VISIBLE_DEVICES=3 python3 -m sglang.launch_server --model $MODEL_NAME --log-requests --log-requests-level 3 --port 8003 &
CUDA_VISIBLE_DEVICES=4 python3 -m sglang.launch_server --model $MODEL_NAME --log-requests --log-requests-level 3 --port 8004 &
CUDA_VISIBLE_DEVICES=5 python3 -m sglang.launch_server --model $MODEL_NAME --log-requests --log-requests-level 3 --port 8005 &
CUDA_VISIBLE_DEVICES=6 python3 -m sglang.launch_server --model $MODEL_NAME --log-requests --log-requests-level 3 --port 8006 &
CUDA_VISIBLE_DEVICES=7 python3 -m sglang.launch_server --model $MODEL_NAME --log-requests --log-requests-level 3 --port 8007 &

echo "All servers started. Ports: 8000-8007"
echo "Press Ctrl+C to stop all servers"

# Wait for all background processes
wait