"""
Benchmark module for attribute-based sorting evaluation.
"""

from typing import Dict, Any, Callable, List
from src.attribute_sorting import AttributeSorting

def run_attribute_benchmark(
    llm_sort_fn: Callable[[str], str],
    size: int = 10,
    items_per_question: tuple = (4, 8),
    categories: List[str] = None,
    seed: int = 42
) -> dict:
    """
    Run benchmark for attribute-based sorting tasks.
    
    Args:
        llm_sort_fn: Function that takes a question string and returns a sorted list as a string
        size: Number of test cases to generate
        items_per_question: Range of items to include in each question
        categories: List of categories to use, or None for all
        seed: Random seed for reproducibility
        
    Returns:
        dict: Benchmark results including:
            - overall_binary_score: Average binary score across all test cases
            - overall_kendall_tau: Average Kendall Tau correlation across all test cases
            - test_cases: List of individual test case results
    """
    # Initialize the attribute sorting task generator
    sorter = AttributeSorting(seed=seed)
    
    # Generate dataset
    dataset = sorter.generate_dataset(
        size=size,
        items_per_question=items_per_question,
        categories=categories
    )
    
    results = {
        "overall_binary_score": 0.0,
        "overall_kendall_tau": 0.0,
        "test_cases": []
    }
    
    total_binary_score = 0.0
    total_kendall_tau = 0.0
    
    for i, entry in enumerate(dataset):
        # Get LLM's answer for the question
        model_answer = llm_sort_fn(entry["question"])
        
        # Evaluate the answer
        eval_result = sorter.evaluate_answer(
            model_answer=model_answer,
            correct_answer=entry["answer"],
            metadata=entry["metadata"]
        )
        
        results["test_cases"].append({
            "id": i,
            "category": entry["metadata"]["category"],
            "direction": entry["metadata"]["direction"],
            "question": entry["question"],
            "correct_answer": entry["answer"],
            "model_answer": model_answer,
            "model_parsed_names": eval_result["model_names"],
            "correct_names": eval_result["correct_names"],
            "binary_score": eval_result["binary_score"],
            "kendall_tau": eval_result["kendall_tau"],
            "kendall_p_value": eval_result["kendall_p_value"]
        })
        
        total_binary_score += eval_result["binary_score"]
        total_kendall_tau += eval_result["kendall_tau"]
    
    if size > 0:
        results["overall_binary_score"] = total_binary_score / size
        results["overall_kendall_tau"] = total_kendall_tau / size
    
    return results 