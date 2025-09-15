export WANDB_API_KEY=2b60f655a687ad1161d31f0002256865e1ace428
export WANDB_PROJECT=llm-reasoning
export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
MODEL=Qwen/Qwen3-235B-A22B-FP8
MODEL_ARGS="model_name=$MODEL,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95,returns_logits:false},use_chat_template=true,tensor_parallel_size=$NUM_GPUS,enable_thinking=true"
OUTPUT_DIR=/mnt/task_wrapper/user_output/artifacts/$MODEL-thinking
mkdir -p $OUTPUT_DIR

## AIME 2024
#TASK=aime24
#lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
#    --output-dir $OUTPUT_DIR --save-details  2>&1 | tee "logs/log_$(date +%Y%m%d_%H%M%S)_${RANDOM}.log"

# AIME 2025
TASK=aime25
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --output-dir $OUTPUT_DIR --save-details  2>&1 | tee "logs/log_$(date +%Y%m%d_%H%M%S)_${RANDOM}.log"

# MATH
TASK=math_500
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --output-dir $OUTPUT_DIR --save-details  2>&1 | tee "logs/log_$(date +%Y%m%d_%H%M%S)_${RANDOM}.log"

# GSM8K
TASK=gsm8k
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --output-dir $OUTPUT_DIR --save-details  2>&1 | tee "logs/log_$(date +%Y%m%d_%H%M%S)_${RANDOM}.log"

# GPQA Diamond
TASK=gpqa:diamond
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --output-dir $OUTPUT_DIR --save-details  2>&1 | tee "logs/log_$(date +%Y%m%d_%H%M%S)_${RANDOM}.log"