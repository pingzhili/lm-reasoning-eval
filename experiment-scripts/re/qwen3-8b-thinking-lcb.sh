export WANDB_API_KEY=2b60f655a687ad1161d31f0002256865e1ace428
export WANDB_PROJECT=llm-reasoning
export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
MODEL=Qwen/Qwen3-8B
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:30000,temperature:0.6,top_p:0.95,returns_logits:false},use_chat_template=true,tensor_parallel_size=$NUM_GPUS,enable_thinking=true"
OUTPUT_DIR=data/evals/$MODEL-thinking
mkdir -p $OUTPUT_DIR

# LCB code gen
TASK=lcb:codegeneration
lighteval vllm $MODEL_ARGS "extended|$TASK|0|0" \
    --output-dir $OUTPUT_DIR --save-details 2>&1 | tee "logs/log_$(date +%m%d_%H%M%S)_$TASK.log"