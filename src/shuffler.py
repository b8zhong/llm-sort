"""
Module for shuffling items in various ways for sorting evaluation.
"""

import random
from typing import List, Dict, Any, TypeVar, Callable, Optional

T = TypeVar("T")


class Shuffler:
    """
    Class for shuffling items in different ways to test sorting algorithms.
    Supports shuffling both numeric lists and lists of dictionaries with attributes.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the shuffler with an optional random seed.

        Args:
            seed: Random seed for reproducibility.
        """
        self.random = random.Random(seed)

    def random_shuffle(self, items: List[T]) -> List[T]:
        """
        Randomly shuffle a list of items.

        Args:
            items: List of items to shuffle

        Returns:
            A new list with shuffled items
        """
        shuffled_items = items.copy()
        self.random.shuffle(shuffled_items)
        return shuffled_items

    def reverse(self, items: List[T]) -> List[T]:
        """
        Reverse a list of items.

        Args:
            items: List of items to reverse

        Returns:
            A new list with items in reverse order
        """
        return items.copy()[::-1]

    def almost_sorted(self, items: List[T], swap_fraction: float = 0.2) -> List[T]:
        """
        Create an almost sorted list by swapping a fraction of adjacent items.

        Args:
            items: List of items to shuffle
            swap_fraction: Fraction of items to swap (default: 0.2)

        Returns:
            A new list with items almost sorted
        """
        shuffled_items = items.copy()
        n = len(shuffled_items)
        num_swaps = max(1, int(n * swap_fraction))

        for _ in range(num_swaps):
            i = self.random.randint(0, n - 2)
            shuffled_items[i], shuffled_items[i + 1] = (
                shuffled_items[i + 1],
                shuffled_items[i],
            )

        return shuffled_items

    def shuffle_with_fixed_positions(
        self, items: List[T], fixed_positions: List[int]
    ) -> List[T]:
        """
        Shuffle items while keeping some positions fixed.

        Args:
            items: List of items to shuffle
            fixed_positions: List of positions (0-indexed) to keep fixed

        Returns:
            A new list with items shuffled except at fixed positions
        """
        shuffled_items = items.copy()
        n = len(shuffled_items)

        # Extract items at fixed positions
        fixed_items = {
            pos: shuffled_items[pos] for pos in fixed_positions if 0 <= pos < n
        }

        # Get movable items
        movable_items = [
            item for i, item in enumerate(shuffled_items) if i not in fixed_positions
        ]

        # Shuffle movable items
        self.random.shuffle(movable_items)

        # Reconstruct the list
        result = []
        movable_idx = 0

        for i in range(n):
            if i in fixed_items:
                result.append(fixed_items[i])
            else:
                result.append(movable_items[movable_idx])
                movable_idx += 1

        return result

    def shuffle_attribute_items(
        self, items: List[Dict[str, Any]], key: str
    ) -> List[Dict[str, Any]]:
        """
        Shuffle a list of dictionaries while preserving the association between
        each item and its attributes.

        Args:
            items: List of dictionaries to shuffle
            key: Key to use for checking if attributes are properly preserved

        Returns:
            A new list with shuffled dictionaries
        """
        return self.random_shuffle(items)

    def create_shuffle_strategies(self) -> Dict[str, Callable[[List[T]], List[T]]]:
        """
        Create a dictionary of different shuffling strategies.

        Returns:
            Dictionary mapping strategy names to shuffling functions
        """
        return {
            "random": self.random_shuffle,
            "reverse": self.reverse,
            "almost_sorted": self.almost_sorted,
            "identity": lambda x: x.copy(),  # No shuffling, just returns a copy
        }

    def apply_strategy(self, items: List[T], strategy: str, **kwargs) -> List[T]:
        """
        Apply a specific shuffling strategy to a list of items.

        Args:
            items: List of items to shuffle
            strategy: Name of the strategy to apply
            **kwargs: Additional arguments to pass to the strategy function

        Returns:
            A new list with shuffled items according to the strategy
        """
        strategies = self.create_shuffle_strategies()

        if strategy not in strategies:
            raise ValueError(
                f"Unknown shuffling strategy: {strategy}. "
                f"Available strategies: {list(strategies.keys())}"
            )

        strategy_fn = strategies[strategy]

        if strategy == "almost_sorted" and "swap_fraction" in kwargs:
            return self.almost_sorted(items, kwargs["swap_fraction"])

        return strategy_fn(items)
