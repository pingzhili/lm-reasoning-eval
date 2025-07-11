export PYTHONPATH="/mnt/task_runtime/lm-reasoning-eval/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_API_KEY=2b60f655a687ad1161d31f0002256865e1ace428
export WANDB_PROJECT=llm-reasoning
export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM

MODEL=Qwen/Qwen2.5-32B
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95},use_chat_template=true,tensor_parallel_size=2"
OUTPUT_DIR=data/evals/$MODEL
WANDB_NAME="qwen-2.5-32b"

# AIME 2024
TASK=aime24
python -m lighteval.main_vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR --save-details

# MATH
TASK=math_500
python -m lighteval.main_vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR --save-details

# GPQA Diamond
TASK=gsm8k
python -m lighteval.main_vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR --save-details
