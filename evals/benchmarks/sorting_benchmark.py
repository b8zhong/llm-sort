"""
Benchmark module for sorting evaluation.
"""

from typing import List
import random

def generate_test_cases() -> List[List]:
    """
    Generate test cases for sorting evaluation.
    
    Returns:
        List of test cases with varying complexity
    """
    test_cases = [
        # Simple numeric lists
        [4, 2, 1, 3],
        [10, -5, 0, 15, 8],
        
        # Lists with duplicates
        [3, 1, 3, 2, 1],
        
        # Edge cases
        [],  # Empty list
        [1],  # Single element
        [1, 1, 1],  # All same elements
        
        # TODO: Add more complex test cases
    ]
    return test_cases

def run_benchmark(evaluator) -> dict:
    """
    Run sorting benchmark using the provided evaluator.
    
    Args:
        evaluator: Instance of SortEvaluator
        
    Returns:
        dict: Benchmark results and metrics
    """
    test_cases = generate_test_cases()
    results = {}
    
    for i, test_case in enumerate(test_cases):
        result = evaluator.evaluate_sorting(test_case)
        results[f"test_case_{i}"] = result
    
    return results 