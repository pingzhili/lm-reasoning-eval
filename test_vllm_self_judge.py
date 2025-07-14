#!/usr/bin/env python3
"""Test VLLM self-judging thinking feature"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["VLLM_USE_V1"] = "0"  # Use legacy engine to avoid compilation issues
os.environ["VLLM_TORCH_COMPILE_LEVEL"] = "0"

from lighteval.models.vllm.vllm_model import VLLMModel, VLLMModelConfig
from lighteval.tasks.requests import Doc

def test_self_judge_model():
    print("Creating VLLMModelConfig with self-judging enabled...")
    
    # Create a simpler config for testing
    config = VLLMModelConfig(
        model_name="Qwen/Qwen3-8B",
        dtype="bfloat16",
        max_model_length=4096,
        gpu_memory_utilization=0.7,
        max_num_seqs=32,  # Reduce for testing
        max_num_batched_tokens=1024,  # Reduce for testing
        use_chat_template=True,
        enable_thinking=True,
        self_judge_thinking=True,
    )
    
    print("Config created successfully!")
    print(f"  enable_thinking: {config.enable_thinking}")
    print(f"  self_judge_thinking: {config.self_judge_thinking}")
    
    print("\nLoading VLLM model...")
    try:
        model = VLLMModel(config)
        print("Model loaded successfully!")
        
        # Create test docs
        test_docs = [
            Doc(
                query="What is 5 + 3?",
                choices=["7", "8", "9"],
                gold_index=1,
                task_name="test",
                instruction="",
                unconditioned_query="",
                num_asked_few_shots=0,
                num_effective_few_shots=0,
                original_query="What is 5 + 3?",
                id="test_simple",
                num_samples=1,
                generation_size=100,
                stop_sequences=None,
            ),
            Doc(
                query="Prove that the square root of 2 is irrational.",
                choices=["proof"],
                gold_index=0,
                task_name="test",
                instruction="",
                unconditioned_query="",
                num_asked_few_shots=0,
                num_effective_few_shots=0,
                original_query="Prove that the square root of 2 is irrational.",
                id="test_complex",
                num_samples=1,
                generation_size=200,
                stop_sequences=None,
            ),
        ]
        
        print("\nTesting self-judging thinking...")
        responses = model.greedy_until_self_judge(test_docs)
        
        print("\nResults:")
        for doc, response in zip(test_docs, responses):
            print(f"\nQuestion: {doc.query}")
            print(f"Response: {response.text[0] if response.text else 'No response'}")
            
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"\nError loading model: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'model' in locals():
            print("\nCleaning up...")
            model.cleanup()

if __name__ == "__main__":
    test_self_judge_model()