for gpu in {4..7}; do
    budget=$((2048 + (gpu - 4) * 2048))
    CUDA_VISIBLE_DEVICES=$gpu bash experiment-scripts/re/qwen3-8b-thinking-budget-lcb.sh $budget &
    echo "Started GPU $gpu with budget $budget"
done