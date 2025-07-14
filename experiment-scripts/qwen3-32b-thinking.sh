export CUDA_VISIBLE_DEVICES=4,5
export WANDB_API_KEY=2b60f655a687ad1161d31f0002256865e1ace428
export WANDB_PROJECT=llm-reasoning
export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM

MODEL=Qwen/Qwen3-32B
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95,returns_logits:false},use_chat_template=true,tensor_parallel_size=2,enable_thinking=true"
OUTPUT_DIR=data/evals/$MODEL-thinking
mkdir -p $OUTPUT_DIR

# AIME 2024
TASK=aime24
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --output-dir $OUTPUT_DIR --save-details  2>&1 | tee "logfile_$(date +%Y-%m-%d_%H-%M-%S).log"

# MATH
TASK=math_500
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --output-dir $OUTPUT_DIR --save-details  2>&1 | tee "logfile_$(date +%Y-%m-%d_%H-%M-%S).log"

# GPQA Diamond
TASK=gsm8k
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --output-dir $OUTPUT_DIR --save-details  2>&1 | tee "logfile_$(date +%Y-%m-%d_%H-%M-%S).log"
