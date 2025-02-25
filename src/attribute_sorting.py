"""
Module for generating and evaluating attribute-based sorting tasks.
"""

import random
from typing import List, Dict, Any, Tuple
import re
import numpy as np
from scipy.stats import kendalltau


class AttributeSorting:
    """Class for generating and evaluating attribute-based sorting tasks."""

    # Different categories of items that can be sorted
    CATEGORIES = {
        "products": {
            "name_prefix": "Product",
            "attribute_name": "price",
            "format_string": "{name}: ${value:.2f}",
            "prompt": "Sort these products by their price in {direction} order:",
        },
        "students": {
            "name_prefix": "Student",
            "attribute_name": "score",
            "format_string": "{name}: {value:.1f}",
            "prompt": "Sort these students by their test score in {direction} order:",
        },
        "restaurants": {
            "name_prefix": "Restaurant",
            "attribute_name": "rating",
            "format_string": "{name}: {value:.1f} stars",
            "prompt": "Sort these restaurants by their rating in {direction} order:",
        },
        "cities": {
            "name_prefix": "City",
            "attribute_name": "population",
            "format_string": "{name}: {value:,} people",
            "prompt": "Sort these cities by their population in {direction} order:",
        },
    }

    def __init__(self, seed: int = 42):
        """
        Initialize the attribute sorting task generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.random = random.Random(seed)

    def generate_dataset(
        self,
        size: int = 10,
        items_per_question: Tuple[int, int] = (4, 8),
        categories: List[str] = None,
    ) -> List[Dict]:
        """
        Generate a dataset of attribute sorting questions.

        Args:
            size: Number of questions to generate
            items_per_question: Range (min, max) of items per question
            categories: List of categories to use, or None for all

        Returns:
            List of dictionaries containing questions and answers
        """
        if categories is None:
            categories = list(self.CATEGORIES.keys())
        else:
            # Validate categories
            for cat in categories:
                if cat not in self.CATEGORIES:
                    raise ValueError(
                        f"Unknown category: {cat}. Available categories: {list(self.CATEGORIES.keys())}"
                    )

        dataset = []
        for _ in range(size):
            # Randomly select a category
            category = self.random.choice(categories)
            category_info = self.CATEGORIES[category]

            # Randomly decide if sorting is ascending or descending
            direction = self.random.choice(["ascending", "descending"])

            # Generate items
            num_items = self.random.randint(*items_per_question)
            items = []

            for i in range(num_items):
                # Generate a value appropriate for the category
                if category == "products":
                    value = round(self.random.uniform(1.99, 999.99), 2)
                elif category == "students":
                    value = round(self.random.uniform(0, 100), 1)
                elif category == "restaurants":
                    value = round(self.random.uniform(1.0, 5.0), 1)
                elif category == "cities":
                    value = self.random.randint(10000, 10000000)

                name = f"{category_info['name_prefix']} {chr(65 + i)}"  # A, B, C, etc.
                item_str = category_info["format_string"].format(name=name, value=value)

                items.append({"name": name, "value": value, "text": item_str})

            # Create the question
            prompt = category_info["prompt"].format(direction=direction)
            question = prompt + "\n" + "\n".join(item["text"] for item in items)

            # Sort items to get the answer
            if direction == "ascending":
                sorted_items = sorted(items, key=lambda x: x["value"])
            else:  # descending
                sorted_items = sorted(items, key=lambda x: x["value"], reverse=True)

            # Format the answer
            answer = [item["name"] for item in sorted_items]
            answer_str = str(answer)

            dataset.append(
                {
                    "question": question,
                    "answer": answer_str,
                    "metadata": {
                        "category": category,
                        "direction": direction,
                        "items": items,
                        "sorted_items": sorted_items,
                    },
                }
            )

        return dataset

    def evaluate_answer(
        self, model_answer: str, correct_answer: str, metadata: Dict
    ) -> Dict:
        """
        Evaluate the model's answer against the correct answer.

        Args:
            model_answer: The model's answer string
            correct_answer: The correct answer string
            metadata: Question metadata

        Returns:
            Dictionary with evaluation metrics
        """
        # Extract names from the model's answer
        model_names = self._extract_names(model_answer, metadata["category"])

        # Extract names from the correct answer
        correct_names = [item["name"] for item in metadata["sorted_items"]]

        # Calculate binary score (1.0 if completely correct, 0.0 otherwise)
        binary_score = 1.0 if model_names == correct_names else 0.0

        # Calculate Kendall Tau correlation
        try:
            # Create numeric indices for each name
            name_to_index = {name: i for i, name in enumerate(correct_names)}

            # Convert model's answer to indices, handling unknown names
            model_indices = []
            for name in model_names:
                if name in name_to_index:
                    model_indices.append(name_to_index[name])

            # If model provided enough valid names, calculate Kendall Tau
            if len(model_indices) > 1:
                correct_indices = list(range(len(correct_names)))
                tau, p_value = kendalltau(model_indices, correct_indices)

                # Handle NaN values
                if np.isnan(tau):
                    tau = 1.0 if binary_score == 1.0 else 0.0
            else:
                tau = 0.0
                p_value = 1.0

        except Exception as e:
            print(f"Error calculating Kendall Tau: {e}")
            tau = 0.0
            p_value = 1.0

        return {
            "binary_score": binary_score,
            "kendall_tau": tau,
            "kendall_p_value": p_value,
            "model_names": model_names,
            "correct_names": correct_names,
        }

    def _extract_names(self, answer_str: str, category: str) -> List[str]:
        """
        Extract item names from the answer string.

        Args:
            answer_str: The answer string
            category: The category of items

        Returns:
            List of item names
        """
        # Get the name prefix for this category
        prefix = self.CATEGORIES[category]["name_prefix"]

        # Use regex to find all occurrences of the prefix followed by a letter
        pattern = rf"{prefix}\s+[A-Z]"
        matches = re.findall(pattern, answer_str)

        # Clean up the matches
        names = [match.strip() for match in matches]

        return names
