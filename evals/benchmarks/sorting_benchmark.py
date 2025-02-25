"""
Benchmark module for sorting evaluation using reasoning-gym.
"""

from typing import Dict, Any, Callable
from src.sort_evaluator import SortEvaluator

def run_benchmark(
    llm_sort_fn: Callable[[str], list],
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
            - overall_score: Average score across all test cases
            - test_cases: List of individual test case results
    """
    evaluator = SortEvaluator(config)
    evaluator.initialize_dataset(size=size, seed=seed)
    
    results = {
        "overall_score": 0.0,
        "test_cases": []
    }
    
    total_score = 0.0
    
    for i, entry in enumerate(evaluator.dataset):
        # Get LLM's answer for the question
        model_answer = llm_sort_fn(entry["question"])
        
        # Evaluate the answer
        result = evaluator.evaluate_sorting(model_answer, entry)
        results["test_cases"].append({
            "id": i,
            "question": entry["question"],
            **result
        })
        
        total_score += result["score"]
    
    results["overall_score"] = total_score / size
    return results 