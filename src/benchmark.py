from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabulate import tabulate

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config


@dataclass
class BenchmarkRow:
    """Represents a row in the benchmark evaluation table."""

    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read benchmark conversation datasets from JSON on disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0, 0.5, or 1 depending on how many expected facts appear.

    A 3-level tier preserves partial matches ('đúng một nửa') instead of counting as total failure.
    """
    if not expected:
        return 1.0
    ans_lower = answer.lower()
    matches = sum(1 for exp in expected if exp.lower() in ans_lower)
    if matches == len(expected):
        return 1.0
    elif matches > 0:
        return 0.5
    return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight quality score for evaluation.

    Considers fact recall, structured clarity, and conciseness.
    Penalizes amnesia responses ('chưa có thông tin').
    """
    if not answer or not answer.strip():
        return 0.0

    ans_lower = answer.lower()
    if "chưa có thông tin" in ans_lower or "rất tiếc" in ans_lower:
        return 0.35

    rec = recall_points(answer, expected)
    format_bonus = 0.0
    if len(answer) >= 15:
        format_bonus += 0.15
    if "\n" in answer or "-" in answer or "." in answer:
        format_bonus += 0.15

    quality = (rec * 0.7) + format_bonus
    return round(min(1.0, quality), 2)


def run_agent_benchmark(
    agent_name: str,
    agent: BaselineAgent | AdvancedAgent,
    conversations: list[dict[str, Any]],
    config: LabConfig,
) -> BenchmarkRow:
    """Evaluate one agent over conversations and recall queries."""
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    for conv in conversations:
        user_id = conv["user_id"]
        conv_id = conv["id"]
        turns = conv.get("turns", [])
        recall_questions = conv.get("recall_questions", [])

        # Step 1: Run conversation turns in a single thread
        thread_id = f"thread_{conv_id}"
        for turn in turns:
            agent.reply(user_id=user_id, thread_id=thread_id, message=turn)

        # Step 2: Query recall questions in fresh threads
        for q_idx, rq in enumerate(recall_questions):
            recall_thread = f"recall_{conv_id}_{q_idx}"
            question = rq["question"]
            expected = rq.get("expected_contains", [])

            result = agent.reply(user_id=user_id, thread_id=recall_thread, message=question)
            answer = result.get("reply", "")

            rec = recall_points(answer, expected)
            qual = heuristic_quality(answer, expected)

            recall_scores.append(rec)
            quality_scores.append(qual)

    # Compute aggregate metrics
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    agent_tokens = agent.token_usage()
    prompt_tokens = agent.prompt_token_usage()
    compactions = agent.compaction_count()

    # Memory growth: size of User.md files in state
    memory_growth = 0
    if isinstance(agent, AdvancedAgent):
        profiles_dir = config.state_dir / "profiles"
        if profiles_dir.exists():
            for p in profiles_dir.rglob("*.md"):
                if p.is_file():
                    memory_growth += p.stat().st_size

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=memory_growth,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark rows as a clean Markdown table."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table_data = []
    for r in rows:
        table_data.append(
            [
                r.agent_name,
                f"{r.agent_tokens_only:,}",
                f"{r.prompt_tokens_processed:,}",
                f"{r.recall_score * 100:.1f}%",
                f"{r.response_quality * 100:.1f}%",
                f"{r.memory_growth_bytes:,}",
                r.compactions,
            ]
        )
    return tabulate(table_data, headers=headers, tablefmt="github")


def run_benchmark_suite(
    suite_name: str,
    dataset_path: Path,
    config: LabConfig,
) -> None:
    """Run both Baseline and Advanced agents on a dataset and print comparative results."""
    print(f"\n=======================================================")
    print(f"  {suite_name}")
    print(f"=======================================================")
    print(f"Dataset: {dataset_path.name}")

    conversations = load_conversations(dataset_path)

    # Clean state before baseline
    shutil.rmtree(config.state_dir / "profiles", ignore_errors=True)
    (config.state_dir / "profiles").mkdir(parents=True, exist_ok=True)
    baseline_agent = BaselineAgent(config, force_offline=True)
    row_baseline = run_agent_benchmark("Baseline", baseline_agent, conversations, config)

    # Clean state before advanced
    shutil.rmtree(config.state_dir / "profiles", ignore_errors=True)
    (config.state_dir / "profiles").mkdir(parents=True, exist_ok=True)
    advanced_agent = AdvancedAgent(config, force_offline=True)
    row_advanced = run_agent_benchmark("Advanced", advanced_agent, conversations, config)

    table = format_rows([row_baseline, row_advanced])
    print("\n" + table + "\n")


def main() -> None:
    """Run Standard Benchmark and Long-Context Stress Benchmark."""
    config = load_config(Path(__file__).resolve().parent.parent)

    std_dataset = config.data_dir / "conversations.json"
    stress_dataset = config.data_dir / "advanced_long_context.json"

    # 1. Standard Benchmark
    run_benchmark_suite("Standard Benchmark", std_dataset, config)

    # 2. Long-Context Stress Benchmark
    run_benchmark_suite("Long-Context Stress Benchmark", stress_dataset, config)


if __name__ == "__main__":
    main()
