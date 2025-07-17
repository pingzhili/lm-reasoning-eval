#!/usr/bin/env python3
"""Test hf_overrides parsing"""

from lighteval.models.utils import ModelConfig

# Test the MODEL_ARGS from the bash script
model_args = "model_name=Qwen/Qwen3-32B,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95,returns_logits:false},use_chat_template=true,tensor_parallel_size=2,enable_thinking=true,hf_overrides={num_experts_per_tok:4}"

print("Testing MODEL_ARGS parsing with hf_overrides...")
print(f"Input: {model_args}")

try:
    parsed_args = ModelConfig._parse_args(model_args)
    print("\nParsed args:")
    for k, v in parsed_args.items():
        print(f"  {k}: {v}")
    
    # Try to create the config
    from lighteval.models.vllm.vllm_model import VLLMModelConfig
    config = VLLMModelConfig.from_args(model_args)
    print("\nSuccess! VLLMModelConfig created:")
    print(f"  model_name: {config.model_name}")
    print(f"  hf_overrides: {config.hf_overrides}")
    print(f"  generation_parameters: {config.generation_parameters}")
    
except Exception as e:
    print(f"\nError: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()