export WANDB_API_KEY=2b60f655a687ad1161d31f0002256865e1ace428
export WANDB_PROJECT=llm-reasoning
export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM

MOE_TOPK=${1:-8}
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
MODEL=Qwen/Qwen3-32B
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95,returns_logits:false},use_chat_template=true,tensor_parallel_size=$NUM_GPUS,enable_thinking=true,hf_overrides={num_experts_per_tok:$MOE_TOPK}"
OUTPUT_DIR=data/evals/$MODEL-moe-topk
mkdir -p $OUTPUT_DIR

# AIME 2024
TASK=aime24
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --output-dir $OUTPUT_DIR --save-details  2>&1 | tee "log_moe_topk_$(date +%Y%m%d_%H%M%S)_${RANDOM}.log"

# MATH
TASK=math_500
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --output-dir $OUTPUT_DIR --save-details  2>&1 | tee "log_moe_topk_$(date +%Y%m%d_%H%M%S)_${RANDOM}.log"