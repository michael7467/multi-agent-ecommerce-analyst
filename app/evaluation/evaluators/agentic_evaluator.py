from __future__ import annotations

import json
from pathlib import Path

from app.agents.planning_agent import PlanningAgent
from app.agents.langgraph_orchestrator import LangGraphOrchestrator
from app.logging.logger import get_logger
from app.evaluation.eval_history import save_eval_run
from app.evaluation.metrics.agentic_metrics import (
    expand_expected_flags,
    tool_call_accuracy,
    tool_call_precision_recall_f1,
    route_exact_match,
    count_active_agents,
    count_active_agents_from_output,
    goal_field_coverage,
)
from app.evaluation.metrics.reliability_metrics import outcome_rate

logger = get_logger("evaluation.agentic")


class AgenticEvaluator:
    def __init__(self) -> None:
        self.planning_agent = PlanningAgent()
        # Lazy, same reasoning as GenerationEvaluator: LangGraphOrchestrator
        # constructs ~19 agents (a real Qdrant connection among them).
        # evaluate_routing() alone only needs planning_agent -- confirmed
        # this was being paid for unnecessarily when a routing-only eval
        # run showed Qdrant cleanup noise despite never touching retrieval.
        self._orchestrator = None

    @property
    def orchestrator(self) -> LangGraphOrchestrator:
        if self._orchestrator is None:
            self._orchestrator = LangGraphOrchestrator()
        return self._orchestrator

    def evaluate_routing(self, query: str, expected_true: list[str]) -> dict:
        """Route correctness / tool-call accuracy / F1. Only needs a
        query, not a product_id -- calls PlanningAgent alone, not the
        full pipeline. One LLM call (the planning step's own), not the
        up-to-16 a full orchestrator run could cost.

        Note the LLM-based portion of planning is not fully deterministic
        -- only the rule-boost layer is (see planning_agent.py's _RULES).
        A predicted flag not in expected_true isn't automatically wrong;
        it may be the planning LLM reasonably adding something the rules
        don't strictly require. tool_call_recall (did we get everything
        required) is the more trustworthy half of this than
        tool_call_precision for that reason.
        """
        expected = expand_expected_flags(expected_true)

        try:
            result = self.planning_agent.run(query)
            predicted = result["plan"]
        except Exception:
            logger.error(f"Planning failed for query={query!r}", exc_info=True)
            return {"query": query, "failed": True}

        f1_data = tool_call_precision_recall_f1(predicted, expected)

        return {
            "query": query,
            "failed": False,
            "predicted_true": sorted(f for f, v in predicted.items() if v),
            "expected_true": sorted(expected_true),
            "tool_call_accuracy": tool_call_accuracy(predicted, expected),
            "tool_call_precision": f1_data["precision"],
            "tool_call_recall": f1_data["recall"],
            "tool_call_f1": f1_data["f1"],
            "route_exact_match": route_exact_match(predicted, expected),
            "n_agents_planned": count_active_agents(predicted),
        }

    def evaluate_goal(self, product_id: str, query: str, expected_true: list[str], top_k: int = 3) -> dict:
        """Agent goal accuracy. Runs the full pipeline (needs a real
        product_id) -- checks whether the final output actually delivered
        the fields the expected flags imply, not just whether the plan
        said it would.
        """
        try:
            result = self.orchestrator.run(product_id=product_id, query=query, top_k=top_k)
            final_output = result["final_output"]
        except Exception:
            logger.error(
                f"Orchestrator run failed for product_id={product_id}, query={query!r}",
                exc_info=True,
            )
            return {"product_id": product_id, "query": query, "failed": True}

        coverage = goal_field_coverage(final_output, expected_true)

        return {
            "product_id": product_id,
            "query": query,
            "failed": False,
            "goal_accuracy": coverage["goal_accuracy"],
            "covered": coverage["covered"],
            "missing": coverage["missing"],
            "cascading_missing": coverage["cascading_missing"],
            "failed_steps": final_output.get("failed_steps", []),
            "n_agents_active": count_active_agents_from_output(final_output),
        }

    def run(self, eval_file: str | Path, top_k: int = 3, include_goal: bool = False) -> dict:
        """eval_file entries: {"query": ..., "expected_true": [...]},
        optionally with "product_id" if include_goal=True (goal accuracy
        needs a real product to run the full pipeline against; routing
        checks don't).
        """
        path = Path(eval_file)
        if not path.exists():
            raise FileNotFoundError(f"Evaluation file not found: {eval_file}")

        with open(path, "r", encoding="utf-8") as f:
            cases = json.load(f)

        routing_results = [
            self.evaluate_routing(case["query"], case["expected_true"])
            for case in cases
        ]

        goal_results = []
        if include_goal:
            for case in cases:
                if "product_id" not in case:
                    logger.warning(f"Skipping goal accuracy for query={case['query']!r}: no product_id given")
                    continue
                goal_results.append(
                    self.evaluate_goal(case["product_id"], case["query"], case["expected_true"], top_k=top_k)
                )

        def avg(results: list[dict], key: str) -> float | None:
            values = [r[key] for r in results if not r["failed"]]
            return sum(values) / len(values) if values else None

        summary = {
            "n_routing_cases": len(routing_results),
            "routing_failed_rate": outcome_rate(routing_results, "failed", True),
            "avg_tool_call_accuracy": avg(routing_results, "tool_call_accuracy"),
            "avg_tool_call_precision": avg(routing_results, "tool_call_precision"),
            "avg_tool_call_recall": avg(routing_results, "tool_call_recall"),
            "avg_tool_call_f1": avg(routing_results, "tool_call_f1"),
            "route_exact_match_rate": (
                sum(1 for r in routing_results if not r["failed"] and r["route_exact_match"])
                / len([r for r in routing_results if not r["failed"]])
                if any(not r["failed"] for r in routing_results) else None
            ),
            "avg_n_agents_planned": avg(routing_results, "n_agents_planned"),
        }

        if include_goal:
            summary.update({
                "n_goal_cases": len(goal_results),
                "goal_failed_rate": outcome_rate(goal_results, "failed", True),
                "avg_goal_accuracy": avg(goal_results, "goal_accuracy"),
                "avg_n_agents_active": avg(goal_results, "n_agents_active"),
            })

        save_eval_run("agentic", summary)

        print(f"\n=== Agentic Evaluation ({len(routing_results)} routing cases) ===")
        for r in routing_results:
            print(r)
        if include_goal:
            print(f"\n=== Goal Accuracy ({len(goal_results)} cases) ===")
            for r in goal_results:
                print(r)
        print("\n=== Summary ===")
        for k, v in summary.items():
            print(f"{k}: {v}")

        return {"routing_results": routing_results, "goal_results": goal_results, "summary": summary}


def run_agentic_eval(
    eval_file: str = "data/eval/agentic_eval_set.json",
    top_k: int = 3,
    include_goal: bool = False,
) -> dict:
    return AgenticEvaluator().run(eval_file, top_k=top_k, include_goal=include_goal)


if __name__ == "__main__":
    run_agentic_eval(include_goal=False)