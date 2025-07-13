#!/usr/bin/env python3
"""
Debug script to understand exactly what vLLM returns for logprobs.
"""
import sys
sys.path.append('src')
import tempfile
import os

# Set minimal environment
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

from vllm import LLM, SamplingParams

# Create a minimal test
llm = LLM(model="microsoft/DialoGPT-medium", max_model_len=512, gpu_memory_utilization=0.3)

# Request logprobs like lighteval does
sampling_params = SamplingParams(
    max_tokens=5,
    temperature=0.6,
    top_p=0.95,
    logprobs=10  # Same as lighteval: sampling_params.logprobs = 10 if returns_logits else 0
)

# Generate
prompts = ["Hello, my name is"]
outputs = llm.generate(prompts, sampling_params)

print("=== vLLM Raw Output Analysis ===")
for i, output in enumerate(outputs):
    print(f"\nOutput {i}:")
    print(f"  Prompt: {output.prompt}")
    print(f"  Generated text: {output.outputs[0].text}")
    print(f"  Number of outputs: {len(output.outputs)}")
    
    for j, completion in enumerate(output.outputs):
        print(f"\n  Completion {j}:")
        print(f"    Token IDs: {completion.token_ids}")
        print(f"    Logprobs type: {type(completion.logprobs)}")
        print(f"    Logprobs length: {len(completion.logprobs) if completion.logprobs else 'None'}")
        
        if completion.logprobs:
            for k, token_logprobs in enumerate(completion.logprobs):
                print(f"\n    Token {k} logprobs:")
                print(f"      Type: {type(token_logprobs)}")
                print(f"      Length: {len(token_logprobs) if token_logprobs else 'None'}")
                
                if token_logprobs:
                    total_entries = len(token_logprobs)
                    none_entries = sum(1 for lp in token_logprobs.values() if lp is None)
                    valid_entries = total_entries - none_entries
                    
                    print(f"      Total entries: {total_entries}")
                    print(f"      None entries: {none_entries}")  
                    print(f"      Valid entries: {valid_entries}")
                    print(f"      Percentage None: {none_entries/total_entries*100:.1f}%")
                    
                    print(f"      Sample entries:")
                    sample_count = 0
                    for idx, lp in token_logprobs.items():
                        if sample_count >= 5:
                            break
                        print(f"        {idx}: {lp}")
                        sample_count += 1
                        
                    if none_entries > 0:
                        print(f"      Sample None entries:")
                        none_count = 0
                        for idx, lp in token_logprobs.items():
                            if lp is None and none_count < 3:
                                print(f"        {idx}: {lp}")
                                none_count += 1
                                
                    # Test my filtering
                    filtered = {str(idx): {"logprob": lp.logprob, "rank": lp.rank, "decoded_token": lp.decoded_token}
                               for idx, lp in token_logprobs.items() 
                               if lp is not None and hasattr(lp, 'logprob') and lp.logprob is not None}
                    print(f"      After my filtering: {len(filtered)} entries")
                    
                    if k >= 1:  # Only show details for first 2 tokens
                        break