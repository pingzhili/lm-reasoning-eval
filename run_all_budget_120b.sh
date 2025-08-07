CUDA_VISIBLE_DEVICES=0 bash experiment-scripts/gptoss-120b-thinking.sh low 8000 &
CUDA_VISIBLE_DEVICES=1 bash experiment-scripts/gptoss-120b-thinking.sh medium 8001 &
CUDA_VISIBLE_DEVICES=2 bash experiment-scripts/gptoss-120b-thinking.sh high 8002 &

sleep 60

CUDA_VISIBLE_DEVICES=3 bash experiment-scripts/gptoss-120b-thinking.sh low 8003 &
CUDA_VISIBLE_DEVICES=4 bash experiment-scripts/gptoss-120b-thinking.sh medium 8004 &
CUDA_VISIBLE_DEVICES=5 bash experiment-scripts/gptoss-120b-thinking.sh high 8005 &