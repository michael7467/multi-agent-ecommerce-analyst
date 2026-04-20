from app.services.buy_decision_service import BuyDecisionService


def test_buy_decision_recommended_for_positive_product():
    service = BuyDecisionService()

    analysis_result = {
        "title": "Sample Headphones",
        "price": 59.99,
        "predicted_class": "high",
        "sentiment": {
            "avg_sentiment_score": 0.90,
            "positive_review_ratio": 0.95,
        },
        "aspect_sentiment": {
            "sound_quality": {"label": "mixed"},
            "battery_life": {"label": "positive"},
            "comfort": {"label": "positive"},
            "build_quality": {"label": "mixed"},
            "price_value": {"label": "positive"},
        },
        "evidence": [{"x": 1}],
        "recommendations": [{"product_id": "abc"}],
    }

    result = service.make_decision(analysis_result)

    assert result["decision"] in {"recommended", "conditionally recommended"}
    assert isinstance(result["pros"], list)
    assert isinstance(result["cons"], list)
    assert "summary" in result