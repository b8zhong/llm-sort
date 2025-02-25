"""
Script to run the attribute-based sorting evaluation.
"""

from evals.benchmarks.attribute_sorting_benchmark import run_attribute_benchmark
from .llm_client import LLMClient
import numpy as np

def main():

    use_openrouter = False
    llm = LLMClient(use_openrouter=use_openrouter)

    print("\n=== Attribute Sorting Evaluation Setup ===")
    print(f"Using API: {'OpenRouter' if use_openrouter else 'OpenAI'}")
    print(f"Model: {llm.sampling_params['model']}")
    print("=" * 80)

    config = {
        "size": 4,  
        "items_per_question": (4, 6),  
        "categories": ["products", "students", "restaurants", "cities"],  
        "seed": 42  
    }

    print("\n=== Task Configuration ===")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 80)

    print("\nRunning attribute sorting evaluation...")
    results = run_attribute_benchmark(
        llm_sort_fn=llm.sort_numbers,
        **config
    )

    print(f"\n=== Results ===")
    print(f"Overall Binary Score: {results['overall_binary_score']:.2%}")
    print(f"Overall Kendall Tau: {results['overall_kendall_tau']:.4f}")
    print("\nDetailed Results:")
    print("-" * 80)

    for case in results['test_cases']:
        print(f"Test Case {case['id']} ({case['category']}, {case['direction']} order):")
        print(f"Question:\n{case['question']}")
        print(f"\nCorrect Names: {case['correct_names']}")
        print(f"Model Parsed Names: {case['model_parsed_names']}")
        print(f"Raw Model Answer: {case['model_answer']}")
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