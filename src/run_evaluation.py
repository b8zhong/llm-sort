"""
Script to run the number sorting evaluation.
"""

from evals.benchmarks.sorting_benchmark import run_benchmark
from .llm_client import LLMClient
import numpy as np

def main():
    use_openrouter = False
    llm = LLMClient(use_openrouter=use_openrouter)

    print("\n=== Evaluation Setup ===")
    print(f"Using API: {'OpenRouter' if use_openrouter else 'OpenAI'}")
    print(f"Model: {llm.sampling_params['model']}")
    print("=" * 80)

    config = {
        'min_numbers': 5,
        'max_numbers': 10,
        'min_decimals': 10,
        'max_decimals': 15,
        'min_value': 10000.0,
        'max_value': 10001.0
    }

    print("\n=== Task Configuration ===")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 80)

    print("\nRunning evaluation...")
    results = run_benchmark(
        llm_sort_fn=llm.sort_numbers,
        config=config,
        size=2,  
        seed=42    
    )

    print(f"\n=== Results ===")
    print(f"Overall Binary Score: {results['overall_binary_score']:.2%}")
    print(f"Overall Kendall Tau: {results['overall_kendall_tau']:.4f}")
    print("\nDetailed Results:")
    print("-" * 80)

    for case in results['test_cases']:
        print(f"Test Case {case['id']}:")
        print(f"Question: {case['question']}")
        print(f"Model Answer: {case['model_answer']}")
        print(f"Correct Answer: {case['correct_answer']}")
        print(f"Binary Score: {case['binary_score']}")
        print(f"Kendall Tau: {case['kendall_tau']:.4f} (p-value: {case['kendall_p_value']:.4f})")

        tau = case['kendall_tau']
        if tau > 0.9:
            interpretation = "Perfect/Near-perfect sorting"
        elif tau > 0.7:
            interpretation = "Strong agreement"
        elif tau > 0.5:
            interpretation = "Moderate agreement"
        elif tau > 0.3:
            interpretation = "Weak agreement"
        elif tau > -0.3:
            interpretation = "Little to no correlation"
        elif tau > -0.5:
            interpretation = "Weak disagreement"
        elif tau > -0.7:
            interpretation = "Moderate disagreement"
        elif tau > -0.9:
            interpretation = "Strong disagreement"
        else:
            interpretation = "Complete/Near-complete reversal"

        print(f"Interpretation: {interpretation}")
        print("-" * 80)

if __name__ == "__main__":
    main()
