from app.agents.planning_agent import PlanningAgent


def test_planning_agent_detects_buy_query():
    agent = PlanningAgent()
    result = agent.run("Should I buy this product for comfort and battery life?")
    plan = result["plan"]

    assert plan["use_data"] is True
    assert plan["use_buy_decision"] is True
    assert plan["use_report"] is True