# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LightEval** is a Python package for evaluating Large Language Models (LLMs) across multiple backends including transformers, VLLM, TGI, and Nanotron. This is a fork/variant focused on LM reasoning evaluation experiments, with additional tools for analyzing model outputs and logits.

## Common Development Commands

### Installation and Setup
```bash
# Install base package
pip install -e .

# Install with development dependencies
pip install -e .[dev]

# Install specific backends
pip install -e .[vllm]          # VLLM backend
pip install -e .[nanotron]      # Nanotron backend
pip install -e .[extended_tasks] # Extended task support
```

### Code Quality and Formatting
```bash
# Format code and fix linting issues
make style

# Check code quality without fixing
make quality

# Manual commands
ruff format .
ruff check --fix .

# Type checking (if available)
# Note: This project doesn't have mypy configured, but uses ruff for type checking
ruff check . --select=ANN,TCH

# IMPORTANT: Always run these commands after implementing new features
# to ensure code quality and catch potential issues
```

### Testing
```bash
# Run fast tests only (default)
pytest tests/

# Run all tests including slow integration tests
pytest tests/ --runslow

# Run specific test categories
pytest tests/metrics/          # Metric tests
pytest tests/models/           # Model tests
pytest tests/tasks/            # Task tests

# Run specific test file
pytest tests/test_unit_base_metrics.py
```

### Running Evaluations
```bash
# Accelerate backend (CPU/GPU with transformers)
lighteval accelerate "model_name=gpt2" "leaderboard|truthfulqa:mc|0|0"

# VLLM backend (fast GPU inference)
lighteval vllm "model_name=meta-llama/Llama-2-7b-hf" "leaderboard|hellaswag|5|1"

# Endpoint backends (TGI, OpenAI, etc.)
lighteval endpoint tgi "model_name=http://localhost:8080" "leaderboard|arc:easy|25|1"
```

## Code Architecture

### Core Components

1. **Entry Points** (`src/lighteval/__main__.py`): Typer-based CLI with subcommands for different backends
2. **Pipeline** (`src/lighteval/pipeline.py`): Core evaluation pipeline orchestrating model loading, task execution, and metric computation
3. **Models** (`src/lighteval/models/`): Abstracted model interfaces for different backends
4. **Tasks** (`src/lighteval/tasks/`): Task definitions, prompts, and request handling
5. **Metrics** (`src/lighteval/metrics/`): Evaluation metrics and scoring functions

### Model Backends Architecture

The project supports multiple model backends through a common interface:

- **Transformers** (`models/transformers/`): HuggingFace transformers integration
- **VLLM** (`models/vllm/`): High-performance inference engine
- **Nanotron** (`models/nanotron/`): Distributed training/inference
- **Endpoints** (`models/endpoints/`): TGI, Inference Endpoints, OpenAI API
- **Custom** (`models/custom/`): User-defined model implementations

### Task System

- **Registry** (`tasks/registry.py`): Central task registration and discovery
- **Templates** (`tasks/templates/`): Reusable task templates (multichoice, continuation, etc.)
- **Default Tasks** (`tasks/default_tasks.py`): Built-in evaluation tasks
- **Extended Tasks** (`tasks/extended/`): Specialized tasks (MT-Bench, IFEval, etc.)

### Metrics Framework

- **Base Metrics** (`metrics/metrics.py`): Core metric implementations
- **Dynamic Metrics** (`metrics/dynamic_metrics.py`): Runtime metric loading
- **LLM-as-Judge** (`metrics/llm_as_judge.py`): Model-based evaluation metrics

## Project-Specific Tools

### Experiment Scripts
- `experiment-scripts/`: Shell scripts for running specific model evaluations
- `tools/plot-token-logits.py`: Utility for analyzing token-level logits

### Data Analysis
- `data/evals/`: Stored evaluation results and outputs
- `plots/`: Generated visualization outputs (HTML logit plots)

### Custom Evaluation Tasks
The repository includes community tasks and extensions:
- `community_tasks/`: Domain-specific evaluation tasks
- Task files in `examples/tasks/`: Task group configurations

## Dependencies and Environment

### Core Dependencies
- **Python 3.10+** required
- **PyTorch 2.0+** for model inference
- **Transformers 4.51.0+** for HuggingFace models
- **Datasets 3.5.0+** for data loading

### Optional Backend Dependencies
- **VLLM 0.8.4+** for fast inference (`pip install lighteval[vllm]`)
- **Nanotron** for distributed evaluation (`pip install lighteval[nanotron]`)
- **LiteLLM** for API providers (`pip install lighteval[litellm]`)

### Math and Reasoning Dependencies
- **SymPy 1.12** for symbolic math evaluation
- **latex2sympy2_extended** for LaTeX math processing
- **word2number** for number parsing

## Configuration

### Model Configuration
Model parameters can be specified via:
1. Command-line key=value pairs: `"model_name=gpt2,dtype=float16"`
2. YAML config files: `examples/model_configs/*.yaml`

### Task Configuration
Tasks are specified as: `"task_type|task_name|num_shots|batch_size"`
- Example: `"leaderboard|arc:easy|25|1"`
- Task lists in: `examples/tasks/*.txt`

## Development Notes

### Code Style
- Uses **Ruff** for formatting and linting (line length: 119)
- Double quotes for strings, 2 lines after imports
- Pre-commit hooks enforce code quality

### Testing Strategy
- **Fast tests**: Unit tests for individual components (default)
- **Slow tests**: Integration tests requiring model loading (`--runslow`)
- **Reference tests**: Validation against known scores
- Mock objects (`FakeModel`) for testing without real models

### Logging and Tracking
- Uses **colorlog** for colored console output
- **WandB** integration for experiment tracking (optional)
- Evaluation results saved to local files, S3, or HuggingFace Hub