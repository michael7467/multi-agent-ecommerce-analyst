from __future__ import annotations

# User-turn prompt templates: the per-request, variable half of each
# call. Untrusted content (review text) is wrapped in clear XML-style
# tags -- OpenAI's own prompt-engineering guide recommends exactly this
# for marking where one piece of content begins and ends -- and the
# corresponding system prompt tells the model explicitly to treat that
# tagged content as data, not instructions.
#
# Kept in their own file, not as inline f-strings buried in service
# methods -- versioned and reviewable on their own, the same way code
# changes are, rather than invisible until someone opens the specific
# method that happens to contain them.


def build_aspect_sentiment_user_prompt(aspect: str, evidence_text: str) -> str:
    aspect_label = aspect.replace("_", " ")

    return (
        f"Aspect: {aspect_label}\n\n"
        f"<review_evidence>\n{evidence_text}\n</review_evidence>"
    )
