from __future__ import annotations

import pandas as pd
import streamlit as st

from app.agents.orchestrator import Orchestrator


st.set_page_config(
    page_title="Multi-Agent E-commerce AI Analyst",
    page_icon="🛍️",
    layout="wide",
)


@st.cache_resource
def load_orchestrator() -> Orchestrator:
    return Orchestrator()


def render_header() -> None:
    st.title("🛍️ Multi-Agent E-commerce AI Analyst")
    st.markdown(
        """
        Analyze an e-commerce product using:

        - **Forecast Agent** for price-class prediction  
        - **Sentiment Agent** for customer perception analysis  
        - **Retrieval Agent** for review-based evidence  
        - **Report Agent** for LLM-generated explanation  
        - **Guardrail Agent** for consistency checking
        """
    )


def render_sidebar() -> tuple[str, str, int, bool]:
    st.sidebar.header("Analysis Settings")

    product_id = st.sidebar.text_input(
        "Product ID",
        value="B09SPZPDJK",
        help="Enter the product_id used in the dataset.",
    )

    query = st.sidebar.text_area(
        "Analysis Query",
        value="sound quality and noise cancellation",
        help="Describe what kind of evidence you want the system to retrieve.",
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


def render_prediction_card(output: dict) -> None:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Predicted Class", str(output.get("predicted_class", "")).upper())

    with col2:
        price = output.get("price")
        st.metric("Product Price", f"${price:.2f}" if isinstance(price, (int, float)) else "N/A")

    with col3:
        guardrail_status = output.get("guardrail_status", "unknown")
        st.metric("Guardrail Status", guardrail_status.upper())


def render_product_info(output: dict) -> None:
    st.subheader("Product Information")
    st.write(f"**Product ID:** {output.get('product_id', '')}")
    st.write(f"**Title:** {output.get('title', '')}")
    st.write(f"**Categories:** {output.get('categories', '')}")


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


def render_report(output: dict) -> None:
    st.subheader("LLM Explanation")
    st.write(output.get("report", ""))


def render_evidence(output: dict) -> None:
    st.subheader("Retrieved Evidence")

    evidence = output.get("evidence", [])
    if not evidence:
        st.info("No evidence retrieved.")
        return

    for i, ev in enumerate(evidence, start=1):
        title = ev.get("review_title", "") or f"Evidence {i}"
        score = ev.get("score", 0.0)

        with st.expander(f"Evidence {i} — Score: {score:.4f}"):
            st.write(f"**Review Title:** {title}")
            st.write(f"**Product Title:** {ev.get('title', '')}")
            st.write(f"**Categories:** {ev.get('categories', '')}")
            st.write("**Review Text:**")
            st.write(ev.get("review_text", ""))


def render_raw_output(output: dict) -> None:
    st.subheader("Raw Output")
    st.json(output)


def main() -> None:
    render_header()

    product_id, query, top_k, show_raw_output = render_sidebar()

    st.markdown("---")

    if st.button("Analyze Product", type="primary"):
        try:
            orchestrator = load_orchestrator()

            with st.spinner("Running multi-agent analysis..."):
                result = orchestrator.run(
                    product_id=product_id,
                    query=query,
                    top_k=top_k,
                )

            output = result["final_output"]

            render_prediction_card(output)
            st.markdown("---")
            render_product_info(output)
            st.markdown("---")
            render_sentiment(output)
            st.markdown("---")
            render_report(output)
            st.markdown("---")
            render_aspect_summaries(output)
            st.markdown("---")
            render_evidence(output)
            st.markdown("---")
            render_recommendations(output)
            if show_raw_output:
                st.markdown("---")
                render_raw_output(output)

        except Exception as e:
            st.error(f"Analysis failed: {e}")
def render_recommendations(output: dict) -> None:
    st.subheader("Similar Product Recommendations")

    recommendations = output.get("recommendations", [])
    if not recommendations:
        st.info("No recommendations available.")
        return

    for i, item in enumerate(recommendations, start=1):
        with st.expander(f"Recommendation {i} — Similarity: {item['similarity_score']:.4f}"):
            st.write(f"**Product ID:** {item.get('product_id', '')}")
            st.write(f"**Title:** {item.get('title', '')}")
            st.write(f"**Categories:** {item.get('categories', '')}")
            price = item.get("price", None)
            st.write(f"**Price:** ${price:.2f}" if isinstance(price, (int, float)) else "**Price:** N/A")
            st.write(f"**Predicted Class:** {str(item.get('predicted_class', '')).upper()}")
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
        with st.expander(label):
            st.write(payload.get("summary", ""))
if __name__ == "__main__":
    main()