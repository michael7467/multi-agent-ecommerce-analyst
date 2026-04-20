from __future__ import annotations

import pandas as pd
import streamlit as st

from app.agents.dynamic_orchestrator import DynamicOrchestrator
from app.observability.metrics import setup_metrics
from app.observability.tracing import setup_tracing

from app.core.config import settings
setup_tracing()
setup_metrics(port=settings.metrics_port)

st.set_page_config(
    page_title="Multi-Agent E-commerce AI Analyst",
    page_icon="🛍️",
    layout="wide",
)


@st.cache_resource
def load_orchestrator() -> DynamicOrchestrator:
    return DynamicOrchestrator()


def render_header() -> None:
    st.title("🛍️ Multi-Agent E-commerce AI Analyst")
    st.markdown(
        """
        Analyze an e-commerce product using:

        - **Planning Agent** for dynamic query-aware routing  
        - **Forecast Agent** for price-class prediction  
        - **Sentiment Agent** for customer perception analysis  
        - **Retrieval Agent** for review-based evidence  
        - **Summarization Agent** for aspect-based summaries  
        - **Recommender Agent** for similar products  
        - **Image Retrieval Agent** for visually similar products  
        - **Report Agent** for LLM-generated explanations  
        - **Guardrail Agent** for consistency checks  
        - **Critic Agent** for evaluation and critique  
        """
    )


def render_sidebar() -> tuple[str, str, int, bool]:
    st.sidebar.header("Analysis Settings")
    st.sidebar.info(
        "This app uses an LLM-based planner to decide which agents to run for your query."
    )

    product_id = st.sidebar.text_input(
        "Product ID",
        value="B09SPZPDJK",
        help="Enter the product_id used in the dataset.",
    )

    query = st.sidebar.text_area(
        "Analysis Query",
        value="sound quality and noise cancellation",
        help="Ask about sentiment, value, complaints, similar products, visual alternatives, themes, or counterfactuals.",
        height=100,
    )

    top_k = st.sidebar.slider(
        "Top-K Evidence",
        min_value=1,
        max_value=5,
        value=3,
        help="Number of retrieved evidence items to show.",
    )

    show_raw_output = st.sidebar.checkbox("Show raw JSON output", value=False)

    return product_id, query, top_k, show_raw_output


def render_execution_plan(result: dict) -> None:
    st.subheader("Execution Plan")

    plan = result.get("plan", {})
    if not plan:
        st.info("No execution plan available.")
        return

    active_agents = [name for name, enabled in plan.items() if enabled]

    st.write("**Planner selected these capabilities:**")
    for agent in active_agents:
        st.write(f"- {agent}")


def render_prediction_card(output: dict) -> None:
    col1, col2, col3 = st.columns(3)

    with col1:
        predicted_class = output.get("predicted_class")
        st.metric(
            "Predicted Class",
            str(predicted_class).upper() if predicted_class is not None else "N/A",
        )

    with col2:
        price = output.get("price")
        st.metric(
            "Product Price",
            f"${price:.2f}" if isinstance(price, (int, float)) else "N/A",
        )

    with col3:
        guardrail_status = output.get("guardrail_status")
        st.metric(
            "Guardrail Status",
            str(guardrail_status).upper() if guardrail_status is not None else "N/A",
        )


def render_product_info(output: dict) -> None:
    st.subheader("Product Information")
    st.write(f"**Product ID:** {output.get('product_id', '')}")
    st.write(f"**Title:** {output.get('title', '')}")
    st.write(f"**Categories:** {output.get('categories', '')}")


def render_memory(output: dict) -> None:
    st.subheader("Stored Product Memory")

    memory = output.get("memory")
    if not memory:
        st.info("No past memory for this product yet.")
        return

    st.write(f"**Last Predicted Class:** {memory.get('last_predicted_class')}")
    st.write(f"**Average Sentiment:** {memory.get('avg_sentiment')}")
    st.write("**Last Report:**")
    st.write(memory.get("last_report"))


def render_sentiment(output: dict) -> None:
    st.subheader("Sentiment Overview")

    sentiment = output.get("sentiment", {})
    avg_score = float(sentiment.get("avg_sentiment_score", 0.0))
    pos = float(sentiment.get("positive_review_ratio", 0.0))
    neu = float(sentiment.get("neutral_review_ratio", 0.0))
    neg = float(sentiment.get("negative_review_ratio", 0.0))

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Avg Sentiment", f"{avg_score:.3f}")
    with col2:
        st.metric("Positive", f"{pos:.1%}")
    with col3:
        st.metric("Neutral", f"{neu:.1%}")
    with col4:
        st.metric("Negative", f"{neg:.1%}")

    sentiment_df = pd.DataFrame(
        {
            "Sentiment": ["Positive", "Neutral", "Negative"],
            "Ratio": [pos, neu, neg],
        }
    ).set_index("Sentiment")

    st.bar_chart(sentiment_df)


def render_aspect_sentiment(output: dict) -> None:
    st.subheader("Aspect-Based Sentiment")

    aspect_sentiment = output.get("aspect_sentiment", {})
    if not aspect_sentiment:
        st.info("No aspect sentiment available.")
        return

    for aspect, payload in aspect_sentiment.items():
        label = aspect.replace("_", " ").title()
        sentiment_label = payload.get("label", "unknown")
        score = float(payload.get("score", 0.0))
        method = payload.get("method", "")

        st.write(
            f"**{label}**: {sentiment_label.upper()} "
            f"(score={score:.2f}, method={method})"
        )


def render_report(output: dict) -> None:
    st.subheader("LLM Explanation")

    report = output.get("report", "")
    if report:
        st.write(report)
    else:
        st.info("No report available.")


def render_critic_report(output: dict) -> None:
    st.subheader("Critic Agent Evaluation")

    critic_report = output.get("critic_report", "")
    if not critic_report:
        st.info("No critic evaluation available.")
        return

    st.text(critic_report)


def render_aspect_summaries(output: dict) -> None:
    st.subheader("Aspect-Based Review Summaries")

    aspect_summaries = output.get("aspect_summaries", {})
    if not aspect_summaries:
        st.info("No aspect summaries available.")
        return

    aspect_labels = {
        "sound_quality": "Sound Quality",
        "battery_life": "Battery Life",
        "comfort": "Comfort",
        "build_quality": "Build Quality",
        "durability": "Durability",
        "price_value": "Price / Value",
    }

    for aspect, payload in aspect_summaries.items():
        label = aspect_labels.get(aspect, aspect)
        summary = payload.get("summary", "").strip()

        with st.expander(label):
            if summary:
                st.write(summary)
            else:
                st.write("No summary available.")


def render_evidence(output: dict) -> None:
    st.subheader("Retrieved Evidence")

    evidence = output.get("evidence", [])
    if not evidence:
        st.info("No evidence retrieved.")
        return

    for i, ev in enumerate(evidence, start=1):
        title = ev.get("review_title", "") or f"Evidence {i}"
        score = float(ev.get("score", 0.0))

        with st.expander(f"Evidence {i} — Score: {score:.4f}"):
            st.write(f"**Review Title:** {title}")
            st.write(f"**Product Title:** {ev.get('title', '')}")
            st.write(f"**Categories:** {ev.get('categories', '')}")
            st.write("**Review Text:**")
            st.write(ev.get("review_text", ""))


def render_recommendations(output: dict) -> None:
    st.subheader("Similar Product Recommendations")

    recommendations = output.get("recommendations", [])
    if not recommendations:
        st.info("No recommendations available.")
        return

    for i, item in enumerate(recommendations, start=1):
        score = float(item.get("similarity_score", 0.0))
        with st.expander(f"Recommendation {i} — Similarity: {score:.4f}"):
            st.write(f"**Product ID:** {item.get('product_id', '')}")
            st.write(f"**Title:** {item.get('title', '')}")
            st.write(f"**Categories:** {item.get('categories', '')}")
            price = item.get("price")
            st.write(
                f"**Price:** ${price:.2f}"
                if isinstance(price, (int, float))
                else "**Price:** N/A"
            )
            st.write(
                f"**Predicted Class:** {str(item.get('predicted_class', '')).upper()}"
            )


def render_image_similar_products(output: dict) -> None:
    st.subheader("Visually Similar Products")

    items = output.get("image_similar_products", [])
    if not items:
        st.info("No visually similar products found.")
        return

    for i, item in enumerate(items, start=1):
        score = float(item.get("similarity_score", 0.0))
        with st.expander(f"Visual Match {i} — Similarity: {score:.4f}"):
            st.write(f"**Product ID:** {item.get('product_id', '')}")
            st.write(f"**Title:** {item.get('title', '')}")
            st.write(f"**Image URL:** {item.get('image_url', '')}")
            st.write(f"**Image Path:** {item.get('image_path', '')}")


def render_topics(output: dict) -> None:
    st.subheader("Top Themes")

    themes = output.get("top_themes", [])
    if not themes:
        st.info("No topics available.")
        return

    for theme in themes:
        st.write(f"**{theme['topic_name']}** (count={theme['count']})")
        st.write(theme["keywords"])


def render_pain_points(output: dict) -> None:
    st.subheader("Customer Pain Points")

    points = output.get("pain_points", [])
    if not points:
        st.info("No pain points detected.")
        return

    for point in points:
        st.write(f"**{point['topic_name']}**")
        st.write(point["keywords"])


def render_counterfactuals(output: dict) -> None:
    st.subheader("Counterfactual Explanations")

    counterfactuals = output.get("counterfactuals", [])
    if not counterfactuals:
        st.info("No counterfactual explanations available.")
        return

    for i, cf in enumerate(counterfactuals, start=1):
        with st.expander(f"Counterfactual {i}"):
            if cf.get("feature") is None:
                st.write(cf.get("explanation", "No explanation available."))
                continue

            st.write(f"**Feature:** {cf.get('feature', '')}")
            st.write(f"**Original Value:** {cf.get('original_value', '')}")
            st.write(f"**New Value:** {cf.get('new_value', '')}")
            st.write(f"**Original Class:** {cf.get('original_class', '')}")
            st.write(f"**New Class:** {cf.get('new_class', '')}")
            st.write(f"**Change Type:** {cf.get('change_type', '')}")
            st.write(f"**Explanation:** {cf.get('explanation', '')}")


def render_raw_output(output: dict) -> None:
    st.subheader("Raw Output")
    st.json(output)
def render_competitive_analysis(output: dict) -> None:
    st.subheader("Competitive Intelligence")

    data = output.get("competitive_analysis", {})
    if not data:
        st.info("No competitive analysis available.")
        return

    base = data.get("base_product", {})
    competitors = data.get("competitors", [])
    insights = data.get("insights", [])

    st.write("**Base Product**")
    st.write(f"**Title:** {base.get('title', '')}")
    st.write(f"**Price:** {base.get('price', 'N/A')}")
    st.write(f"**Predicted Class:** {str(base.get('predicted_class', '')).upper()}")
    st.write(f"**Average Sentiment:** {base.get('avg_sentiment', 'N/A')}")

    st.markdown("---")
    st.write("**Top Competitors**")

    if not competitors:
        st.info("No competitors available.")
    else:
        for i, comp in enumerate(competitors, start=1):
            with st.expander(f"Competitor {i}: {comp.get('title', '')}"):
                st.write(f"**Product ID:** {comp.get('product_id', '')}")
                st.write(f"**Price:** {comp.get('price', 'N/A')}")
                st.write(f"**Predicted Class:** {str(comp.get('predicted_class', '')).upper()}")
                st.write(f"**Average Sentiment:** {comp.get('avg_sentiment', 'N/A')}")
                st.write(f"**Similarity Score:** {comp.get('similarity_score', 0.0):.4f}")

    st.markdown("---")
    st.write("**Key Insights**")

    if not insights:
        st.info("No competitive insights available.")
    else:
        for insight in insights:
            st.write(f"- {insight}")
def render_buy_decision(output: dict) -> None:
    st.subheader("Should You Buy It?")

    decision = output.get("buy_decision", {})
    if not decision:
        st.info("No buy decision available.")
        return

    st.write(f"**Decision:** {decision.get('decision', '').upper()}")
    st.write(f"**Summary:** {decision.get('summary', '')}")

    pros = decision.get("pros", [])
    cons = decision.get("cons", [])
    recommended_for = decision.get("recommended_for", [])
    not_recommended_for = decision.get("not_recommended_for", [])

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Pros**")
        for item in pros:
            st.write(f"- {item}")

        st.write("**Recommended For**")
        for item in recommended_for:
            st.write(f"- {item}")

    with col2:
        st.write("**Cons**")
        for item in cons:
            st.write(f"- {item}")

        st.write("**Not Recommended For**")
        for item in not_recommended_for:
            st.write(f"- {item}")
def render_trend_analysis(output: dict) -> None:
    st.subheader("Trend Detection")

    trend_data = output.get("trend_analysis", {})
    if not trend_data:
        st.info("No trend analysis available.")
        return

    rising = trend_data.get("rising_categories", [])
    declining = trend_data.get("declining_categories", [])
    complaints = trend_data.get("emerging_complaints", [])
    seasonal = trend_data.get("seasonal_patterns", [])

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Rising Categories**")
        if rising:
            for item in rising:
                st.write(
                    f"- {item['category']} (trend score: {item['trend_score']:.2f})"
                )
        else:
            st.write("No rising categories detected.")

        st.write("**Emerging Complaints**")
        if complaints:
            for item in complaints:
                st.write(
                    f"- {item['complaint']} (trend score: {item['trend_score']:.2f})"
                )
        else:
            st.write("No emerging complaints detected.")

    with col2:
        st.write("**Declining Categories**")
        if declining:
            for item in declining:
                st.write(
                    f"- {item['category']} (trend score: {item['trend_score']:.2f})"
                )
        else:
            st.write("No declining categories detected.")

        st.write("**Seasonal Patterns**")
        if seasonal:
            for item in seasonal:
                st.write(
                    f"- {item['category']} peaks in month {item['peak_month']} "
                    f"(count: {item['peak_review_count']})"
                )
        else:
            st.write("No seasonal patterns detected.")


def main() -> None:
    render_header()
    product_id, query, top_k, show_raw_output = render_sidebar()

    st.markdown("---")

    if st.button("Analyze Product", type="primary"):
        try:
            orchestrator = load_orchestrator()

            with st.spinner("Running dynamic multi-agent analysis..."):
                result = orchestrator.run(
                    product_id=product_id,
                    query=query,
                    top_k=top_k,
                )

            output = result["final_output"]

            render_execution_plan(result)
            st.markdown("---")

            if any(
                key in output for key in ["predicted_class", "price", "guardrail_status"]
            ):
                render_prediction_card(output)
                st.markdown("---")

            if any(key in output for key in ["title", "categories", "product_id"]):
                render_product_info(output)
                st.markdown("---")

            if "memory" in output:
                render_memory(output)
                st.markdown("---")

            if "sentiment" in output:
                render_sentiment(output)
                st.markdown("---")

            if "aspect_sentiment" in output:
                render_aspect_sentiment(output)
                st.markdown("---")

            if "report" in output:
                render_report(output)
                st.markdown("---")

            if "critic_report" in output:
                render_critic_report(output)
                st.markdown("---")

            if "aspect_summaries" in output:
                render_aspect_summaries(output)
                st.markdown("---")

            if "evidence" in output:
                render_evidence(output)
                st.markdown("---")

            if "recommendations" in output:
                render_recommendations(output)
                st.markdown("---")

            if "image_similar_products" in output:
                render_image_similar_products(output)
                st.markdown("---")

            if "top_themes" in output:
                render_topics(output)
                st.markdown("---")

            if "pain_points" in output:
                render_pain_points(output)
                st.markdown("---")

            if "counterfactuals" in output:
                render_counterfactuals(output)
                st.markdown("---")
            if "competitive_analysis" in output:
                render_competitive_analysis(output)
                st.markdown("---")
            if "buy_decision" in output:
                render_buy_decision(output)
                st.markdown("---")
            if "trend_analysis" in output:
                render_trend_analysis(output)
                st.markdown("---")
            if show_raw_output:
                render_raw_output(output)

        except Exception as e:
            st.error(f"Analysis failed: {e}")


if __name__ == "__main__":
    main()