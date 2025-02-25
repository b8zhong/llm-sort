# LLM Sort Evaluation

This project evaluates Large Language Models' (LLMs) ability to perform sorting tasks using the reasoning-gym framework. It specifically focuses on the `number_sorting` task from reasoning-gym.

## Project Structure

- `src/`: Source code for the evaluation framework
  - `sort_evaluator.py`: Main evaluator class using reasoning-gym
- `evals/`: Evaluation scripts and results
  - `benchmarks/`: Benchmark implementation using reasoning-gym tasks

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Use OpenAI or OpenRouter by setting the environment variables:
```bash
touch .env
echo OPENAI_API_KEY=your_api_key >> .env
echo OPENROUTER_API_KEY=your_api_key >> .env
```

## Usage

Here's an example of how to use the evaluator with your LLM:

```python
from evals.benchmarks.sorting_benchmark import run_benchmark

# Define your LLM function that takes a question and returns a sorted list
def my_llm_sort(question: str) -> list:
    # Your LLM implementation here
    # Should return a list of strings representing sorted numbers
    # Example: ['-80', '-72', '-51', '48']
    pass

# Optional configuration for the number_sorting task
config = {
    'min_numbers': 3,
    'max_numbers': 10,
    'min_decimals': 0,
    'max_decimals': 2,
    'min_value': -100.0,
    'max_value': 100.0
}

# Run the benchmark
results = run_benchmark(
    llm_sort_fn=my_llm_sort,
    config=config,
    size=100,  # Number of test cases
    seed=42    # For reproducibility
)

print(f"Overall Score: {results['overall_score']}")
```

The benchmark will evaluate your LLM's ability to:
1. Parse and understand sorting questions
2. Correctly sort numbers in the specified order (ascending/descending)
3. Format the output according to reasoning-gym's requirements

## Task Format

The `number_sorting` task from reasoning-gym generates questions like:

```
Sort these numbers in ascending order: 48, -51, -72, -80
```

Your LLM should return answers in the format:
```python
['-80', '-72', '-51', '48']  # Numbers as strings in a list
```