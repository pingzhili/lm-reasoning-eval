CUDA_VISIBLE_DEVICES=0 bash experiment-scripts/qwen3-8b-self-judge.sh &
CUDA_VISIBLE_DEVICES=1 bash experiment-scripts/qwen3-32b-self-judge.sh &
CUDA_VISIBLE_DEVICES=2,3 bash experiment-scripts/qwen3-235b-self-judge.sh &