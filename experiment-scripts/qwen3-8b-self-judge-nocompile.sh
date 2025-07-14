#!/bin/bash

export CUDA_VISIBLE_DEVICES=7
export WANDB_API_KEY=2b60f655a687ad1161d31f0002256865e1ace428
export WANDB_PROJECT=llm-reasoning
export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM
export VLLM_USE_V1=0  # Use the legacy engine instead of V1
export VLLM_TORCH_COMPILE_LEVEL=0  # Disable torch compilation

MODEL=Qwen/Qwen3-8B
# Enable both thinking capability and self-judging
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=8192,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95,returns_logits:false},use_chat_template=true,enable_thinking=true,self_judge_thinking=true"
OUTPUT_DIR=data/evals/$MODEL-self-judge

# Create output directory
mkdir -p $OUTPUT_DIR

# Test on a small sample of GSM8K for quick testing
TASK=gsm8k
echo "Running $TASK with self-judging thinking..."
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --output-dir $OUTPUT_DIR --save-details --max-samples 10 2>&1 | tee $OUTPUT_DIR/$TASK_output.log

# Also test on a reasoning task
TASK=math_500
echo "Running $TASK with self-judging thinking..."
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --output-dir $OUTPUT_DIR --save-details --max-samples 5 2>&1 | tee $OUTPUT_DIR/${TASK}_output.log

echo "Evaluation complete! Check $OUTPUT_DIR for results."