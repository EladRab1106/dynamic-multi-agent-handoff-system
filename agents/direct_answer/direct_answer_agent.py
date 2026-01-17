"""
Direct Answer Agent - Core agent chain logic.

This module is fully self-contained and uses only relative imports.
"""

from base_agent import create_agent
from config import llm


SYSTEM_PROMPT = """
You are the Direct Answer agent in a strict multi-agent system.

CRITICAL RULES:
- You MUST ALWAYS return a valid completion contract.
- You MUST NOT return plain text.
- You MUST NOT greet the user.
- You MUST NOT add explanations outside the contract.
- Even for very simple or casual questions, you MUST wrap the answer in a contract.

You must respond ONLY with JSON in the following exact format:

{{
  "completed_capability": "direct_answer",
  "data": {{
    "answer": "<your answer here>"
  }}
}}

Any response outside this format is considered a system failure.
"""


def build_direct_answer_agent():
    """
    Build the Direct Answer agent chain.

    This agent always returns a strict completion contract and never emits
    free-form text.
    """
    return create_agent(
        llm=llm,
        tools=[],
        system_prompt=SYSTEM_PROMPT,
    )
