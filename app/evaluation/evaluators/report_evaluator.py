from __future__ import annotations

from app.services.class_alignment import check_class_alignment


def check_report_alignment(predicted_class: str, report: str) -> dict:

    result = check_class_alignment(predicted_class, report)

    return {
        "predicted_class": str(predicted_class).strip().lower(),
        "is_aligned": result["is_aligned"],
        "status": result["status"],
        "reasons": result["reasons"],
        "score": 1.0 if result["is_aligned"] else 0.0,
    }


if __name__ == "__main__":
    sample_report = "The predicted price class for this product is high."
    result = check_report_alignment("high", sample_report)

    print("Report Evaluation Result:")
    print(result)