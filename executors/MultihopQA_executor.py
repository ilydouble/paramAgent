# from typing import Any
# from .executor_types import ExecuteResult, Executor
# from openai import OpenAI  # if not already imported

# class MultiHopQAExecutor(Executor):
#     def execute(self, func: str, tests: list, timeout: int = 5) -> ExecuteResult:
#         # MultihopQA does not use execute in this implementation
#         raise NotImplementedError("MultiHopQAExecutor.execute is not used for evaluation.")

#     def evaluate(self, answer: str, golden_truth: str, timeout: int = 5) -> bool:
#         """
#         Evaluate whether the predicted answer matches the gold answer.

#         Args:
#             answer (str): The predicted answer string.
#             golden_truth (str): The ground-truth answer string.
#             timeout (int): Timeout (unused).

#         Returns:
#             bool: True if answers are considered equivalent (exact match or semantically).
#         """
        
#         # First check exact match
#         if answer.strip().lower() == golden_truth.strip().lower():
#             return True

#         prompt = f"""
#         Determine if the following two answers are semantically similar or equivalent in the context of a factual QA task.
#         Let's not be too strict. For instance: 
#         1) "Aroostook" and "Aroostook County, Maine, United States" should be considered similar;
#         2) Spinning industry and spining should be considered similar.
#         3) Some abbreviations and their full forms should be considered equivalent. For instance, "Britain", "UK", and "United Kingdom" are similar.
#         4) Some fine-grained categories and their corresponding coarse-grained categories should be considered similar. e.g., “musician” and “musical artist” are similar. "music" and "rock music" are similar.

#         Answer 1: {answer}
#         Answer 2: {golden_truth}

#         Respond only with "Yes" or "No".
#         """

#         try:
#             client = OpenAI()
#             response = client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[{"role": "user", "content": prompt.strip()}],
#                 temperature=0.00001,
#                 max_tokens=8,
#             )
#             result = response.choices[0].message.content.strip().lower()
#             print("executor eval:", result)
#             return result.startswith("yes")
#         except Exception:
#             print("executor eval: exception")
#             return False


import os
import re
from typing import Any
from .executor_types import ExecuteResult, Executor
from openai import OpenAI

# Judge model is served through a relay endpoint.
# The API key is read from the OPENAI_API_KEY environment variable.
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "https://www.openai-labs.com/v1")


def _normalize_qa_answer(ans: str) -> str:
    """
    Lowercase, collapse whitespace, strip wrapping quotes/brackets and trailing
    punctuation, so punctuation/space-only differences don't trigger the GPT
    judge (mirrors qa/eval_hotpot_simple.normalize_qa_answer).
    """
    ans = ans.strip().lower()
    ans = re.sub(r"\s+", " ", ans)
    ans = ans.strip('"\'`()[]{}*')
    ans = ans.rstrip(".,;:!?")
    return ans.strip()

class MultiHopQAExecutor(Executor):
    def execute(self, func: str, tests: list, timeout: int = 5) -> ExecuteResult:
        # MultihopQA does not use execute in this implementation
        raise NotImplementedError("MultiHopQAExecutor.execute is not used for evaluation.")

    def evaluate(self, answer: str, golden_truth: str, timeout: int = 5) -> bool:
        """
        Evaluate whether the predicted answer matches the gold answer.

        Args:
            answer (str): The predicted answer string.
            golden_truth (str): The ground-truth answer string.
            timeout (int): Timeout (unused).

        Returns:
            bool: True if answers are considered equivalent (exact match,
            punctuation/whitespace-normalized match, or semantically).
        """

        # First check exact match
        if answer.strip().lower() == golden_truth.strip().lower():
            return True

        # Second check normalized match (case/punctuation/whitespace-insensitive)
        if _normalize_qa_answer(answer) == _normalize_qa_answer(golden_truth):
            return True

        prompt = f"""
        Determine if the following two answers are semantically similar or equivalent in the context of a factual QA task.
        Let's not be too strict. For instance: 
        1) "Aroostook" and "Aroostook County, Maine, United States" should be considered similar;
        2) Spinning industry and spining should be considered similar.
        3) Some abbreviations and their full forms should be considered equivalent. For instance, "Britain", "UK", and "United Kingdom" are similar.
        4) Some fine-grained categories and their corresponding coarse-grained categories should be considered similar. e.g., “musician” and “musical artist” are similar. "music" and "rock music" are similar.

        Answer 1: {answer}
        Answer 2: {golden_truth}

        Respond only with "Yes" or "No".
        """

        try:
            client = OpenAI(
                base_url=JUDGE_BASE_URL,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt.strip()}],
                temperature=0.00001,
                max_tokens=8,
            )
            result = response.choices[0].message.content.strip().lower()
            print("executor eval:", result)
            return result.startswith("yes")
        except Exception:
            print("executor eval: exception")
            return False
