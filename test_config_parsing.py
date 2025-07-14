#!/usr/bin/env python3
"""Test configuration parsing for VLLMModelConfig"""

from lighteval.models.vllm.vllm_model import VLLMModelConfig

# Test MODEL_ARGS from the bash script
model_args = "model_name=Qwen/Qwen3-8B,dtype=bfloat16,max_model_length=8192,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95,returns_logits:false},use_chat_template=true,enable_thinking=true,self_judge_thinking=true"

print("Testing MODEL_ARGS parsing...")
print(f"Input: {model_args}")

try:
    config = VLLMModelConfig.from_args(model_args)
    print("\nSuccess! Config created:")
    print(f"  model_name: {config.model_name}")
    print(f"  enable_thinking: {config.enable_thinking}")
    print(f"  self_judge_thinking: {config.self_judge_thinking}")
    print(f"  generation_parameters: {config.generation_parameters}")
except Exception as e:
    print(f"\nError: {e}")
    print(f"Error type: {type(e).__name__}")
    
    # Try to understand the issue
    from lighteval.models.utils import ModelConfig
    parsed_args = ModelConfig._parse_args(model_args)
    print(f"\nParsed args: {parsed_args}")
    
    # Check if generation_parameters needs special handling
    if "generation_parameters" in parsed_args:
        print(f"\ngeneration_parameters type: {type(parsed_args['generation_parameters'])}")
        print(f"generation_parameters value: {parsed_args['generation_parameters']}")