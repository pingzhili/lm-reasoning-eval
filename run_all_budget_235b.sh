for gpu in {0..3}; do
    budget=$((2048 + gpu * 2048))
    gpu1=$((gpu * 2))
    gpu2=$((gpu * 2 + 1))
    CUDA_VISIBLE_DEVICES=$gpu1,$gpu2 bash experiment-scripts/qwen3-235b-thinking-budget.sh $budget &
    echo "Started GPU $gpu1,$gpu2 with budget $budget"
done

wait

for gpu in {0..3}; do
    budget=$((2048 + gpu * 2048))
    CUDA_VISIBLE_DEVICES=$((gpu * 2)), $((gpu * 2 + 1)) bash experiment-scripts/qwen3-235b-thinking-budget.sh $budget &
    echo "Started GPU $gpu with budget $budget"
done

wait


for gpu in {0..3}; do
    budget=$((2048 + gpu * 2048))
    CUDA_VISIBLE_DEVICES=$((gpu * 2)), $((gpu * 2 + 1)) bash experiment-scripts/qwen3-235b-thinking-budget.sh $budget &
    echo "Started GPU $gpu with budget $budget"
done

wait