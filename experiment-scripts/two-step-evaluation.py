#!/usr/bin/env python3
"""
Example script for running two-step evaluation with thinking judgment.

This script demonstrates how to use the TwoStepEvaluator to:
1. First judge if questions require thinking
2. Then answer with appropriate thinking mode
"""

import json
import argparse
from pathlib import Path
from typing import List

from lighteval.models.vllm.vllm_model import VLLMModel, VLLMModelConfig
from lighteval.models.vllm.two_step_evaluator import TwoStepEvaluator
from lighteval.tasks.lighteval_task import Doc


def create_sample_docs() -> List[Doc]:
    """Create sample documents for testing"""
    samples = [
        {
            "query": "What is 1 + 1?",
            "answer": "2"
        },
        {
            "query": "What is 15 × 23?", 
            "answer": "345"
        },
        {
            "query": "A train travels 120 miles in 2 hours. If it maintains the same speed, how far will it travel in 5 hours?",
            "answer": "300 miles"
        },
        {
            "query": "What is the capital of France?",
            "answer": "Paris"
        },
        {
            "query": "If x + 5 = 12, what is x?",
            "answer": "7"
        },
        {
            "query": "A bakery sells cupcakes for $2 each and cookies for $1 each. If Sarah buys 3 cupcakes and some cookies, and her total bill is $11, how many cookies did she buy?",
            "answer": "5 cookies"
        }
    ]
    
    docs = []
    for i, sample in enumerate(samples):
        doc = Doc(
            task_name="math_sample",
            query=sample["query"],
            choices=[sample["answer"]],
            gold_index=0,
            instruction="",
            target_for_fewshot_sorting=[sample["answer"]],
            kwargs={"expected_answer": sample["answer"]}
        )
        docs.append(doc)
    
    return docs


def main():
    parser = argparse.ArgumentParser(description="Run two-step evaluation")
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct", help="Model name")
    parser.add_argument("--judgment-thinking", action="store_true", 
                       help="Use thinking mode for judgment phase")
    parser.add_argument("--output", default="two_step_results.json", 
                       help="Output file for results")
    
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    print(f"Judgment uses thinking: {args.judgment_thinking}")
    
    # Create model config
    model_config = VLLMModelConfig(
        model=args.model,
        enable_thinking=True,  # This will be overridden per request
        tensor_parallel_size=1,
        max_length=2048
    )
    
    # Initialize model
    model = VLLMModel(model_config)
    
    # Create evaluator
    evaluator = TwoStepEvaluator(
        model=model, 
        judgment_uses_thinking=args.judgment_thinking
    )
    
    # Create sample documents
    docs = create_sample_docs()
    
    print(f"\nRunning evaluation on {len(docs)} sample questions...")
    
    # Run evaluation and comparison
    results = evaluator.evaluate_and_compare(docs)
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print summary
    stats = results["statistics"]
    print(f"\nSummary:")
    print(f"Total questions: {stats['total_questions']}")
    print(f"Judged to need thinking: {stats['judged_need_thinking']}")
    print(f"Judged to not need thinking: {stats['judged_no_thinking']}")
    
    print(f"\nDetailed results:")
    for i, result in enumerate(results["two_step_results"], 1):
        print(f"{i}. {result['question']}")
        print(f"   Judgment: {result['judgment']} (Thinking: {result['thinking_used']})")
        print(f"   Answer: {result['answer'][:100]}{'...' if len(result['answer']) > 100 else ''}")
        print()


if __name__ == "__main__":
    main()