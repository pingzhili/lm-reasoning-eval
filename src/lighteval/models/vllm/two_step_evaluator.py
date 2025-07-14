"""Two-step evaluation module for thinking capability assessment.

This module implements a two-step evaluation process:
1. First asks the model if a question requires thinking
2. Then generates the answer with/without thinking based on the judgment
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from lighteval.models.vllm.vllm_model import VLLMModel, VLLMModelConfig
from lighteval.tasks.lighteval_task import Doc
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.prompt_manager import PromptManager


@dataclass
class TwoStepResult:
    """Result from two-step evaluation"""
    question: str
    thinking_judgment: str
    thinking_required: bool
    final_answer: str
    answer_with_thinking: bool
    judgment_logprobs: Optional[List[float]] = None
    answer_logprobs: Optional[List[float]] = None


class TwoStepEvaluator:
    """Evaluator that performs two-step evaluation with thinking judgment"""
    
    JUDGMENT_PROMPT_TEMPLATE = """You are an expert at determining whether a question requires step-by-step thinking or can be answered directly.

Question: {question}

Your task: Determine if this question requires step-by-step thinking. DO NOT attempt to solve or answer the question itself. Just analyze its complexity.

Consider these factors:
- Simple arithmetic like 1+1, 2×3, or 10-5 does NOT need thinking
- Basic factual recall (e.g., "What is the capital of France?") does NOT need thinking
- Single-step calculations or conversions usually do NOT need thinking
- Multi-step mathematical problems DO need thinking
- Complex word problems DO need thinking
- Problems requiring logical reasoning or multiple considerations DO need thinking
- Problems with multiple constraints or conditions DO need thinking

Respond with ONLY "YES" or "NO":
- YES = This question requires step-by-step thinking
- NO = This question can be answered directly without step-by-step thinking

Your judgment (YES or NO):"""
    
    def __init__(self, model: VLLMModel, judgment_uses_thinking: bool = True):
        """
        Initialize the two-step evaluator.
        
        Args:
            model: The VLLM model to use
            judgment_uses_thinking: Whether to use thinking mode for the judgment step
        """
        self.model = model
        self.judgment_uses_thinking = judgment_uses_thinking
        self.judgment_prompt_manager = PromptManager(
            model.tokenizer,
            task_name="thinking_judgment",
            enable_thinking=judgment_uses_thinking,
            vocab_size=model.tokenizer.vocab_size
        )
        
    def judge_thinking_requirement(self, question: str) -> Tuple[bool, str, Optional[List[float]]]:
        """
        Ask the model if the question requires thinking.
        
        Returns:
            Tuple of (requires_thinking, judgment_response, logprobs)
        """
        judgment_prompt = self.JUDGMENT_PROMPT_TEMPLATE.format(question=question)
        
        # Create a Doc for the judgment
        judgment_doc = Doc(
            task_name="thinking_judgment",
            query=judgment_prompt,
            choices=["YES", "NO"],
            gold_index=0,  # Dummy value
            instruction="",
            target_for_fewshot_sorting=["YES", "NO"],
            kwargs={}
        )
        
        # Get model's judgment
        responses = self.model.greedy_until(
            [judgment_doc],
            max_tokens=10,  # Only need YES/NO
            temperature=0.0,
            stop_sequences=["\n"],
            enable_thinking=self.judgment_uses_thinking
        )
        
        response = responses[0]
        judgment_text = response.result[0].strip().upper()
        
        # Parse judgment
        requires_thinking = self._parse_judgment(judgment_text)
        
        # Extract logprobs if available
        logprobs = None
        if response.result_logprobs and len(response.result_logprobs) > 0:
            logprobs = response.result_logprobs[0]
            
        return requires_thinking, judgment_text, logprobs
    
    def _parse_judgment(self, judgment: str) -> bool:
        """Parse the judgment response to determine if thinking is required"""
        # Clean the response
        judgment = judgment.strip().upper()
        
        # Look for YES/NO in the response
        if "YES" in judgment and "NO" not in judgment:
            return True
        elif "NO" in judgment and "YES" not in judgment:
            return False
        else:
            # Default to True if unclear
            print(f"Warning: Unclear judgment '{judgment}', defaulting to requiring thinking")
            return True
    
    def generate_answer(self, doc: Doc, enable_thinking: bool) -> ModelResponse:
        """Generate answer for the question with specified thinking setting"""
        # Create a new prompt manager with the specified thinking setting
        temp_prompt_manager = PromptManager(
            self.model.tokenizer,
            task_name=doc.task_name,
            enable_thinking=enable_thinking,
            vocab_size=self.model.tokenizer.vocab_size
        )
        
        # Temporarily replace the model's prompt manager
        original_prompt_manager = self.model.prompt_manager
        self.model.prompt_manager = temp_prompt_manager
        
        try:
            # Generate response
            responses = self.model.greedy_until(
                [doc],
                max_tokens=1024,
                temperature=0.0,
                enable_thinking=enable_thinking
            )
            return responses[0]
        finally:
            # Restore original prompt manager
            self.model.prompt_manager = original_prompt_manager
    
    def evaluate_two_step(self, docs: List[Doc]) -> List[TwoStepResult]:
        """
        Perform two-step evaluation on a list of documents.
        
        For each document:
        1. Judge if thinking is required
        2. Generate answer with appropriate thinking setting
        """
        results = []
        
        for i, doc in enumerate(docs):
            print(f"Processing document {i+1}/{len(docs)}...")
            
            # Extract the question from the doc
            question = doc.query
            
            # Step 1: Judge if thinking is required
            requires_thinking, judgment, judgment_logprobs = self.judge_thinking_requirement(question)
            print(f"  Judgment: {judgment} (Thinking required: {requires_thinking})")
            
            # Step 2: Generate answer with appropriate thinking setting
            answer_response = self.generate_answer(doc, enable_thinking=requires_thinking)
            answer = answer_response.result[0] if answer_response.result else ""
            answer_logprobs = answer_response.result_logprobs[0] if answer_response.result_logprobs else None
            
            # Create result
            result = TwoStepResult(
                question=question,
                thinking_judgment=judgment,
                thinking_required=requires_thinking,
                final_answer=answer,
                answer_with_thinking=requires_thinking,
                judgment_logprobs=judgment_logprobs,
                answer_logprobs=answer_logprobs
            )
            
            results.append(result)
            
        return results
    
    def evaluate_and_compare(self, docs: List[Doc]) -> Dict:
        """
        Evaluate documents using both thinking and non-thinking modes,
        plus the two-step approach, for comparison.
        """
        print(f"Running two-step evaluation (judgment uses thinking: {self.judgment_uses_thinking})...")
        two_step_results = self.evaluate_two_step(docs)
        
        print("\nRunning always-thinking evaluation...")
        thinking_results = []
        for doc in docs:
            response = self.generate_answer(doc, enable_thinking=True)
            thinking_results.append(response.result[0] if response.result else "")
        
        print("\nRunning never-thinking evaluation...")
        no_thinking_results = []
        for doc in docs:
            response = self.generate_answer(doc, enable_thinking=False)
            no_thinking_results.append(response.result[0] if response.result else "")
        
        # Compile comparison results
        comparison = {
            "config": {
                "judgment_uses_thinking": self.judgment_uses_thinking
            },
            "two_step_results": [
                {
                    "question": r.question,
                    "judgment": r.thinking_judgment,
                    "thinking_used": r.thinking_required,
                    "answer": r.final_answer
                }
                for r in two_step_results
            ],
            "always_thinking_answers": thinking_results,
            "never_thinking_answers": no_thinking_results,
            "statistics": {
                "total_questions": len(docs),
                "judged_need_thinking": sum(1 for r in two_step_results if r.thinking_required),
                "judged_no_thinking": sum(1 for r in two_step_results if not r.thinking_required)
            }
        }
        
        return comparison