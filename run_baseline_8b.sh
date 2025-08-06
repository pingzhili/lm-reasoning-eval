CUDA_VISIBLE_DEVICES=0 bash experiment-scripts/qwen3-8b-thinking.sh &
CUDA_VISIBLE_DEVICES=1 bash experiment-scripts/qwen3-8b-non-thinking.sh &

sleep 600

CUDA_VISIBLE_DEVICES=2 bash experiment-scripts/qwen3-8b-thinking.sh &
CUDA_VISIBLE_DEVICES=3 bash experiment-scripts/qwen3-8b-non-thinking.sh &