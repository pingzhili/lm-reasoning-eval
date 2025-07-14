# Self-Judging Thinking Implementation

This document describes the implementation of the self-judging thinking feature for LightEval.

## Overview

The self-judging thinking feature allows models to dynamically determine whether a question requires step-by-step thinking before generating an answer. This is a two-step process:

1. **Judgment Step**: The model evaluates whether the question requires careful reasoning
2. **Generation Step**: The model generates an answer with or without thinking based on the judgment

## Implementation Details

### 1. Configuration

Added `self_judge_thinking` parameter to `VLLMModelConfig` in `src/lighteval/models/vllm/vllm_model.py`:

```python
self_judge_thinking: bool = False  # whether to let model self-judge if thinking is needed per question
```

### 2. Core Implementation

Added the following methods to `VLLMModel`:

- `THINKING_JUDGE_TEMPLATE`: A prompt template for asking the model to judge if thinking is needed
- `_create_judge_doc()`: Creates a judgment doc from the original doc
- `_parse_thinking_judgment()`: Parses the YES/NO response from the judgment
- `greedy_until_self_judge()`: The main method that orchestrates the two-step process

### 3. Pipeline Integration

Modified `src/lighteval/pipeline.py` to check for self-judging capability and use the appropriate method:

```python
if hasattr(self.model, '_config') and hasattr(self.model._config, 'self_judge_thinking') and self.model._config.self_judge_thinking:
    model_outputs = self.model.greedy_until_self_judge(docs)
else:
    model_outputs = self.model.greedy_until(docs)
```

### 4. Usage

To enable self-judging thinking, add the parameter to your model configuration:

```bash
MODEL_ARGS="model_name=Qwen/Qwen3-8B,enable_thinking=true,self_judge_thinking=true,..."
```

### 5. Example Script

Created `experiment-scripts/qwen3-8b-self-judge.sh` for testing the feature with Qwen3-8B model.

### 6. Testing

Comprehensive unit tests are provided in `tests/models/vllm/test_self_judge_thinking.py` covering:
- Judge doc creation
- Judgment parsing
- Workflow with/without self-judging enabled
- Prompt manager state restoration

## How It Works

1. When `self_judge_thinking=true`, the model first receives a judgment prompt:
   ```
   Analyze the following question and determine if it requires step-by-step thinking to solve correctly:
   
   Question: {question}
   
   Does this question require careful reasoning or step-by-step thinking? Answer with only 'YES' or 'NO'.
   ```

2. Based on the YES/NO response, the model's `enable_thinking` is temporarily set for that specific question

3. The model then generates the actual answer with or without thinking mode enabled

4. This allows the model to adaptively use its thinking capability only when necessary, potentially improving efficiency

## Limitations

- Currently only supports synchronous VLLM models (async models will fall back to regular generation)
- The judgment adds an extra inference step which may impact overall latency
- The effectiveness depends on the model's ability to accurately judge question complexity