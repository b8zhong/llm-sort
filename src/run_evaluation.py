"""
Script to run the number sorting evaluation.
"""

from evals.benchmarks.sorting_benchmark import run_benchmark
from .llm_client import LLMClient

def main():
    # Initialize the LLM client
    llm = LLMClient(use_openrouter=False)  # Set to False to use OpenAI API
    
    # Configuration for the number_sorting task
    config = {
        'min_numbers': 3,
        'max_numbers': 10,
        'min_decimals': 0,
        'max_decimals': 100,
        'min_value': 10000.0,
        'max_value': 10001.0
    }
    
    # Run the benchmark
    results = run_benchmark(
        llm_sort_fn=llm.sort_numbers,
        config=config,
        size=2,  # Number of test cases
        seed=42    # For reproducibility
    )
    
    # Print results
    print(f"\nOverall Score: {results['overall_score']:.2%}")
    print("\nDetailed Results:")
    print("-" * 80)
    
    for case in results['test_cases']:
        print(f"Test Case {case['id']}:")
        print(f"Question: {case['question']}")
        print(f"Model Answer: {case['model_answer']}")
        print(f"Correct Answer: {case['correct_answer']}")
        print(f"Score: {case['score']}")
        print("-" * 80)

if __name__ == "__main__":
    main() 