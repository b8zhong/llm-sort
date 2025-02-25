"""
LLM client implementation for number sorting evaluation.
"""

import os
from typing import Any, Iterable
import time
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def llm_generate(
    client: OpenAI,
    messages: Iterable[ChatCompletionMessageParam],
    sampling_params: dict[str, Any],
) -> ChatCompletion:
    """
    Generate LLM response with retry logic.
    """
    max_retry = 3
    for trial in range(max_retry):
        try:
            return client.chat.completions.create(
                messages=messages,
                **sampling_params,
            )
        except Exception as e:
            print(f"Failure response (attempt {trial + 1}/{max_retry}):", e)
            time.sleep(trial * trial)  # quadratic backoff
            if trial == max_retry - 1:
                raise


def create_sort_prompt(question: str) -> list[dict]:
    """
    Create a prompt for the sorting task.
    """
    is_attribute_sorting = any(
        phrase in question
        for phrase in [
            "Sort these products",
            "Sort these students",
            "Sort these restaurants",
            "Sort these cities",
        ]
    )

    if is_attribute_sorting:
        system_prompt = """You are a helpful assistant that sorts items by their attributes.
Please follow these instructions carefully:
1. Return ONLY a list of item names in the correct order
2. Format your answer like this example: ['Student C', 'Student A', 'Student B']
3. Do not include any explanations or additional text
4. Make sure to include all items from the question"""
    else:
        system_prompt = """You are a helpful assistant that sorts numbers.
Please follow these instructions carefully:
1. Return ONLY a list of strings representing the sorted numbers
2. Format your answer exactly like this example: ['-80', '-72', '-51', '48']
3. Do not include any explanations or additional text
4. Make sure all numbers are strings inside the list"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


class LLMClient:
    def __init__(self, use_openrouter: bool = True):
        """
        Initialize LLM client with either OpenRouter or OpenAI.

        Args:
            use_openrouter: If True, use OpenRouter API, else use OpenAI API
        """
        if use_openrouter:
            api_key = os.getenv("OPENROUTER_API_KEY")
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                timeout=90.0,
            )
            model = "anthropic/claude-3-sonnet"
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=api_key)
            model = "gpt-4o-mini"

        self.sampling_params = {
            "model": model,
            "max_tokens": 4096,
        }

    def sort_numbers(self, question: str) -> str:
        """
        Process a sorting question and return the sorted list as a string.

        Args:
            question: The sorting question from reasoning-gym or attribute sorting

        Returns:
            str: String representation of the sorted list
        """
        messages = create_sort_prompt(question)
        response = llm_generate(
            client=self.client, messages=messages, sampling_params=self.sampling_params
        )

        # Return the raw string response after cleaning
        if response.choices[0].message.content is None:
            raise ValueError("No response from LLM")
        answer = response.choices[0].message.content.strip()
        return answer.replace("```python", "").replace("```", "").strip()
