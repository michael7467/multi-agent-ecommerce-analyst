from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from app.config.paths import EVAL_HISTORY_DB_PATH


DB_PATH = EVAL_HISTORY_DB_PATH


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS eval_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        evaluator_type TEXT NOT NULL,
        summary_json TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


def save_eval_run(evaluator_type: str, summary: dict) -> None:
    """evaluator_type: one of "retrieval", "agentic", "visual", "generation".
    summary: the exact summary dict each evaluator's run() already produces --
    stored as-is, not normalized into a shared schema, since the four
    evaluators genuinely don't share one (n_cases vs n_routing_cases vs
    n_products). Never raises -- a dashboard write failing should not take
    down the evaluator run that's producing real, useful console output
    regardless of whether this succeeds.
    """
    try:
        init_db()
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO eval_runs (evaluator_type, summary_json, created_at) VALUES (?, ?, ?)",
            (evaluator_type, json.dumps(summary), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        conn.close()
    except Exception:

        pass


def load_eval_history(evaluator_type: str | None = None, limit: int = 100) -> list[dict]:
    """Most recent first. Returns [] if the DB doesn't exist yet (no runs
    saved so far) rather than raising -- a dashboard with zero history yet
    is a normal, expected state, not an error.
    """
    if not DB_PATH.exists():
        return []

    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if evaluator_type:
        cursor.execute(
            "SELECT * FROM eval_runs WHERE evaluator_type = ? ORDER BY created_at DESC LIMIT ?",
            (evaluator_type, limit),
        )
    else:
        cursor.execute(
            "SELECT * FROM eval_runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )

    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        results.append({
            "id": row["id"],
            "evaluator_type": row["evaluator_type"],
            "created_at": row["created_at"],
            "summary": json.loads(row["summary_json"]),
        })
    return results


if __name__ == "__main__":
    init_db()
    print(f"Eval history database initialized at {DB_PATH}")
