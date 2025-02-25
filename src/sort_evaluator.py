"""
Module for evaluating LLM sorting capabilities using reasoning-gym.
"""

import reasoning_gym
from typing import List, Dict, Any

class SortEvaluator:
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the sort evaluator with reasoning-gym.
        
        Args:
            config: Optional configuration for number_sorting task
                   Default values if None:
                   - min_numbers = 3
                   - max_numbers = 10
                   - min_decimals = 0
                   - max_decimals = 2
                   - min_value = -100.0
                   - max_value = 100.0
        """
        self.config = config or {}
        self.dataset = None

    def initialize_dataset(self, size: int = 100, seed: int = 42):
        """
        Initialize a new dataset with given size and seed.
        
        Args:
            size: Number of test cases to generate
            seed: Random seed for reproducibility
        """
        self.dataset = reasoning_gym.create_dataset(
            'number_sorting',
            size=size,
            seed=seed,
            **self.config
        )

    def evaluate_sorting(self, model_answer: List[str], question_entry: Dict) -> dict:
        """
        Evaluate an LLM's sorting answer against reasoning-gym's verification.
        
        Args:
            model_answer: The sorted list of numbers as strings from the LLM
            question_entry: The original question entry from reasoning-gym
            
        Returns:
            dict: Evaluation metrics including:
                - score: 1.0 if correct, 0.0 if incorrect
                - metadata: Original question metadata
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not initialized. Call initialize_dataset first.")

        score = self.dataset.score_answer(
            answer=model_answer,
            entry=question_entry
        )

        return {
            "score": score,
            "metadata": question_entry["metadata"],
            "correct_answer": question_entry["answer"],
            "model_answer": model_answer
        } 