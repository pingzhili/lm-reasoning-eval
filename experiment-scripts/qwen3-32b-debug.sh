export CUDA_VISIBLE_DEVICES=4,5
export WANDB_API_KEY=2b60f655a687ad1161d31f0002256865e1ace428
export WANDB_PROJECT=llm-reasoning
export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM

MODEL=Qwen/Qwen3-32B
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95,returns_logits:true},use_chat_template=true,tensor_parallel_size=2,enable_thinking=false"
OUTPUT_DIR=data/evals/$MODEL-nothinking-debug

# AIME 2024
TASK=aime24
mkdir -p $OUTPUT_DIR/$TASK # for logging
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --output-dir $OUTPUT_DIR --save-details  --max-samples 4 2>&1 | tee $OUTPUT_DIR/$TASK_output.log