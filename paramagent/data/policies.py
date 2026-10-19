"""Online-only policy execution: no imports of verifiers or supervision."""
from __future__ import annotations

from typing import Callable

from .config import ModelConfig
from .schema import DecisionState, ModelCall, StrategyTrace, TaskSample

Invoke = Callable[[str, str, ModelConfig, str, str], ModelCall]


def task_prompt(task: TaskSample) -> tuple[str, str]:
    if task.domain == "code":
        system = "Return a complete Python solution matching the requested interface in a Python code block."
    else:
        system = "Solve the task with concise reasoning. End with exactly one line: FINAL_ANSWER: <answer>."
    user = f"Context:\n{task.context}\n\nQuestion:\n{task.problem}" if task.context else task.problem
    return system, user


def execute_initial(task: TaskSample, actor: ModelConfig, invoke: Invoke) -> DecisionState:
    system, user = task_prompt(task)
    return DecisionState(task, invoke("initial", "actor", actor, system, user))


def retain_initial(state: DecisionState) -> StrategyTrace:
    # No additional model call and no feedback from gold correctness.
    return StrategyTrace("retain_initial_v1", (), state.initial.output)


def execute_diversity(state: DecisionState, actor: ModelConfig,
                      preference: ModelConfig, invoke: Invoke) -> StrategyTrace:
    _, task = task_prompt(state.task)
    guidance = invoke(
        "preference", "preference", preference,
        "Generate diverse, actionable critique and guidance for revising the initial response. "
        "Use only the supplied task and response. Do not claim access to a reference answer or hidden tests.",
        f"Task:\n{task}\n\nInitial response:\n{state.initial.output}",
    )
    system, _ = task_prompt(state.task)
    repaired = invoke(
        "repair", "actor", actor, system,
        f"Task:\n{task}\n\nInitial response:\n{state.initial.output}\n\n"
        f"Diversity guidance:\n{guidance.output}\n\nProduce one revised final response.",
    )
    # Final result is fixed by strategy, never chosen using offline rewards.
    return StrategyTrace("preference_repair_single_v1", (guidance, repaired), repaired.output)
