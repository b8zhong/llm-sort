"""
Benchmark module for sorting evaluation using reasoning-gym.
"""

from typing import Dict, Any, Callable
from src.sort_evaluator import SortEvaluator

def run_benchmark(
    llm_sort_fn: Callable[[str], str],
    config: Dict[str, Any] = None,
    size: int = 100,
    seed: int = 42
) -> dict:
    """
    Run sorting benchmark using reasoning-gym's number_sorting task.

    Args:
        llm_sort_fn: Function that takes a question string and returns a sorted list of strings
        config: Optional configuration for number_sorting task
        size: Number of test cases to generate
        seed: Random seed for reproducibility

    Returns:
        dict: Benchmark results including:
            - overall_binary_score: Average binary score across all test cases
            - overall_kendall_tau: Average Kendall Tau correlation across all test cases
            - test_cases: List of individual test case results
    """
    evaluator = SortEvaluator(config)
    evaluator.initialize_dataset(size=size, seed=seed)

    results = {
        "overall_binary_score": 0.0,
        "overall_kendall_tau": 0.0,
        "test_cases": []
    }

    total_binary_score = 0.0
    total_kendall_tau = 0.0

    for i, entry in enumerate(evaluator.dataset):

        model_answer = llm_sort_fn(entry["question"])

        result = evaluator.evaluate_sorting(model_answer, entry)
        results["test_cases"].append({
            "id": i,
            "question": entry["question"],
            **result
        })

        total_binary_score += result["binary_score"]
        total_kendall_tau += result["kendall_tau"]

    results["overall_binary_score"] = total_binary_score / size
    results["overall_kendall_tau"] = total_kendall_tau / size
    return results