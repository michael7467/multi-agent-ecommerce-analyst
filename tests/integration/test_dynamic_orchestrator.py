from app.agents.dynamic_orchestrator import DynamicOrchestrator


def test_orchestrator_returns_plan_and_output(monkeypatch):
    orchestrator = DynamicOrchestrator()

    monkeypatch.setattr(
        orchestrator.memory_agent,
        "run",
        lambda product_id: {"memory": {}},
    )
    monkeypatch.setattr(
        orchestrator.planning_agent,
        "run",
        lambda query: {
            "plan": {
                "use_data": True,
                "use_sentiment": False,
                "use_aspect_sentiment": False,
                "use_forecast": False,
                "use_retrieval": False,
                "use_recommender": False,
                "use_image_retrieval": False,
                "use_summarization": False,
                "use_topics": False,
                "use_counterfactuals": False,
                "use_competitive": False,
                "use_buy_decision": False,
                "use_trends": False,
                "use_report": False,
                "use_guardrail": False,
                "use_critic": False,
            }
        },
    )
    monkeypatch.setattr(
        orchestrator.data_agent,
        "run",
        lambda product_id: {
            "title": "Test Product",
            "categories": "Electronics",
            "price": 49.99,
        },
    )

    result = orchestrator.run(
        product_id="B09SPZPDJK",
        query="basic query",
        top_k=3,
    )

    assert "plan" in result
    assert "final_output" in result
    assert result["final_output"]["title"] == "Test Product"