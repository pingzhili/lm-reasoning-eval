# Self-Judging Thinking Fix Summary

## Issue Identified

When running the experiment script `experiment-scripts/qwen3-8b-self-judge.sh`, the model loading would fail with a torch compilation error. The root causes were:

1. **Torch Compilation Issues**: The VLLM V1 engine has torch compilation enabled by default which was causing failures during model initialization.

2. **Async Model Support**: The `greedy_until_self_judge` method was only implemented in the synchronous `VLLMModel` class, but not in the `AsyncVLLMModel` class. When the async model was used, it would not have access to the self-judging functionality.

## Fixes Applied

### 1. Async Model Support

Added an async version of `greedy_until_self_judge` to the `AsyncVLLMModel` class:

```python
async def greedy_until_self_judge(self, docs: list[Doc]) -> list[ModelResponse]:
    """
    Async version of two-step generation process:
    1. Ask model if thinking is needed for each question
    2. Generate answer with or without thinking based on judgment
    """
    # Implementation details...
```

### 2. Pipeline Update

Updated the pipeline to properly call the async version of `greedy_until_self_judge` when using async models:

```python
# In pipeline.py _run_model_async method
if hasattr(self.model, '_config') and hasattr(self.model._config, 'self_judge_thinking') and self.model._config.self_judge_thinking:
    model_outputs = await self.model.greedy_until_self_judge(docs)
else:
    model_outputs = await self.model.greedy_until(docs)
```

### 3. Test Coverage

Added unit tests for the async version to ensure proper functionality.

### 4. Environment Configuration

Created an updated script that disables torch compilation to avoid the compilation errors:

```bash
export VLLM_USE_V1=0  # Use the legacy engine instead of V1
export VLLM_TORCH_COMPILE_LEVEL=0  # Disable torch compilation
```

## Updated Files

1. **src/lighteval/models/vllm/vllm_model.py**: Added async `greedy_until_self_judge` method
2. **src/lighteval/pipeline.py**: Updated to support async self-judging
3. **tests/models/vllm/test_self_judge_thinking.py**: Added async test case
4. **experiment-scripts/qwen3-8b-self-judge-nocompile.sh**: Script with compilation disabled

## How to Run

Use the updated script to avoid compilation issues:

```bash
bash experiment-scripts/qwen3-8b-self-judge-nocompile.sh
```

Or set the environment variables manually:

```bash
export VLLM_USE_V1=0
export VLLM_TORCH_COMPILE_LEVEL=0
# Then run your original script
```

## Verification

The configuration parsing works correctly as verified by the test script. The self-judging thinking feature is now fully implemented for both sync and async VLLM models.