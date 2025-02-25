"""
Module for evaluating LLM sorting capabilities using reasoning-gym.
"""

import reasoning_gym
from typing import List, Dict, Any
import numpy as np
from scipy.stats import kendalltau
import re

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

    def _convert_list_string_to_floats(self, list_str: str) -> List[float]:
        """
        Convert a string representation of a list to actual float values.

        Args:
            list_str: String representation of a list, e.g., "['1.23', '4.56']"

        Returns:
            List of float values
        """

        numbers = re.findall(r'-?\d+\.?\d*', list_str)
        return [float(num) for num in numbers]

    def evaluate_sorting(self, model_answer: str, question_entry: Dict) -> dict:
        """
        Evaluate an LLM's sorting answer against reasoning-gym's verification.

        Args:
            model_answer: The sorted list of numbers as a string from the LLM
            question_entry: The original question entry from reasoning-gym

        Returns:
            dict: Evaluation metrics including:
                - score: 1.0 if correct, 0.0 if incorrect (reasoning-gym)
                - kendall_tau: Kendall Tau correlation (-1 to 1)
                - metadata: Original question metadata
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not initialized. Call initialize_dataset first.")

        binary_score = self.dataset.score_answer(
            answer=model_answer,
            entry=question_entry
        )

        try:
            ground_truth_list = self._convert_list_string_to_floats(question_entry["answer"])
            model_list = self._convert_list_string_to_floats(model_answer)
            tau, p_value = kendalltau(ground_truth_list, model_list)

            if np.isnan(tau):
                tau = 1.0 if binary_score == 1.0 else 0.0

        except Exception as e:
            print(f"Error calculating Kendall Tau: {e}")
            tau = 0.0
            p_value = 1.0

        return {
            "binary_score": binary_score,
            "kendall_tau": tau,
            "kendall_p_value": p_value,
            "metadata": question_entry["metadata"],
            "correct_answer": question_entry["answer"],
            "model_answer": model_answer
        }
