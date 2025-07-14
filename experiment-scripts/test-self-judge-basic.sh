#!/bin/bash

export CUDA_VISIBLE_DEVICES=7
export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM
export TORCH_COMPILE_BACKEND=eager # Disable torch compilation for testing

MODEL=Qwen/Qwen3-8B
# Enable both thinking capability and self-judging - simpler config for testing
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=4096,gpu_memory_utilization=0.7,generation_parameters={max_new_tokens:512,temperature:0.0},use_chat_template=true,enable_thinking=true,self_judge_thinking=true"
OUTPUT_DIR=data/evals/$MODEL-self-judge-test

# Create output directory
mkdir -p $OUTPUT_DIR

# Test on a very small sample
TASK=gsm8k
echo "Running $TASK with self-judging thinking (test mode)..."
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --output-dir $OUTPUT_DIR --save-details --max-samples 2 2>&1 | tee $OUTPUT_DIR/test_output.log

echo "Test complete! Check $OUTPUT_DIR for results."