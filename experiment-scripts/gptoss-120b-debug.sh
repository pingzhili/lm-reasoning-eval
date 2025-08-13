export WANDB_API_KEY=2b60f655a687ad1161d31f0002256865e1ace428
export WANDB_PROJECT=llm-reasoning

THINKING_EFFORT=${1-low}
PORT=${2-8000}
BASE_URL=http://localhost:$PORT/v1
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
MODEL=openai/gpt-oss-120b
MODEL_ARGS="model_name=$MODEL,base_url=$BASE_URL,provider=sgl,api_key=\"\",generation_parameters={max_new_tokens:32768,temperature:1.0,top_p:1.0,returns_logits:false,thinking_budget:16,reasoning_effort:\"$THINKING_EFFORT\"}"
OUTPUT_DIR=data/evals/$MODEL-thinking-$THINKING_EFFORT-debug
mkdir -p $OUTPUT_DIR
#
## AIME 2024
#TASK=aime24
#lighteval endpoint litellm $MODEL_ARGS "lighteval|$TASK|0|0" \
#    --output-dir $OUTPUT_DIR --save-details 2>&1 | tee "logs/log_$(date +%m%d_%H%M%S)_$TASK.log"
#
## AIME 2025
#TASK=aime25
#lighteval endpoint litellm $MODEL_ARGS "lighteval|$TASK|0|0" \
#    --output-dir $OUTPUT_DIR --save-details 2>&1 | tee "logs/log_$(date +%m%d_%H%M%S)_$TASK.log"
#
## MATH
#TASK=math_500
#lighteval endpoint litellm $MODEL_ARGS "lighteval|$TASK|0|0" \
#    --output-dir $OUTPUT_DIR --save-details 2>&1 | tee "logs/log_$(date +%m%d_%H%M%S)_$TASK.log"

# GSM8K
TASK=gsm8k
lighteval endpoint litellm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --output-dir $OUTPUT_DIR --save-details 2>&1 | tee "logs/log_$(date +%m%d_%H%M%S)_$TASK.log"

## GPQA Diamond
#TASK=gpqa:diamond
#lighteval endpoint litellm $MODEL_ARGS "lighteval|$TASK|0|0" \
#    --output-dir $OUTPUT_DIR --save-details 2>&1 | tee "logs/log_$(date +%m%d_%H%M%S)_$TASK.log"