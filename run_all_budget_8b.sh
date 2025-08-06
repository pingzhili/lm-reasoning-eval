for gpu in {0..7}; do
    budget=$((1024 + gpu * 1024))
    CUDA_VISIBLE_DEVICES=$gpu bash experiment-scripts/qwen3-8b-thinking-budget.sh $budget &
    echo "Started GPU $gpu with budget $budget"
done