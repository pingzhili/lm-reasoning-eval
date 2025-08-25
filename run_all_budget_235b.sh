for gpu in {0..3}; do
    budget=$((2048 + gpu * 2048))
    CUDA_VISIBLE_DEVICES=$((gpu * 2)), $((gpu * 2 + 1)) bash experiment-scripts/qwen3-30b-thinking-budget.sh $budget &
    echo "Started GPU $gpu with budget $budget"
done

wait

