# Self-Judging Thinking with Metrics Storage

## Overview

When `self_judge_thinking=true` is enabled, the model will judge whether each question requires step-by-step thinking before generating the answer. The judgment result (True/False) is now stored in the evaluation details for analysis.

## How It Works

1. **Judgment Storage**: During the self-judging process, the model stores each judgment in `_self_judgments` dictionary with the doc ID as key:
   ```python
   self._self_judgments[doc.id] = needs_thinking  # True or False
   ```

2. **Metrics Integration**: The pipeline automatically adds the judgment to the metrics output:
   ```python
   if hasattr(self.model, '_self_judgments') and doc.id in self.model._self_judgments:
       output['self_judged_thinking'] = self.model._self_judgments[doc.id]
   ```

3. **Details Storage**: The judgment is saved as part of the metrics in the evaluation details parquet file.

## Usage

### Running with Self-Judging

```bash
MODEL_ARGS="model_name=Qwen/Qwen3-8B,...,enable_thinking=true,self_judge_thinking=true"
lighteval vllm $MODEL_ARGS "lighteval|gsm8k|0|0" --save-details
```

### Analyzing Results

After evaluation, you can load the details parquet file to analyze self-judging patterns:

```python
import pandas as pd

# Load the details
df = pd.read_parquet("data/evals/Qwen3-8B-self-judge/details/details_gsm8k_*.parquet")

# Check self-judgment results
if 'metrics' in df.columns:
    # Extract self_judged_thinking from metrics column
    df['self_judged_thinking'] = df['metrics'].apply(
        lambda x: x.get('self_judged_thinking', None) if isinstance(x, dict) else None
    )
    
    # Analyze judgment patterns
    judgment_counts = df['self_judged_thinking'].value_counts()
    print(f"Questions judged to need thinking: {judgment_counts.get(True, 0)}")
    print(f"Questions judged to NOT need thinking: {judgment_counts.get(False, 0)}")
```

## Example Output

In the saved details, each sample will have a `metrics` field containing:
```json
{
    "exact_match": 0.8,
    "quasi_exact_match": 0.9,
    "self_judged_thinking": true  // <-- The judgment result
}
```

## Benefits

1. **Analysis**: You can analyze which types of questions the model thinks require reasoning
2. **Correlation**: Check if self-judgment correlates with accuracy
3. **Debugging**: Understand model behavior on different question types
4. **Optimization**: Use patterns to improve prompting strategies

## Technical Details

- Judgments are stored per document ID to handle batched evaluations
- Works with both sync and async VLLM models
- No modification needed to existing metrics - judgment is added automatically
- Compatible with all existing evaluation workflows