from __future__ import annotations


def check_report_alignment(predicted_class: str, report: str) -> dict:
    predicted_class = str(predicted_class).strip().lower()
    report_text = str(report).strip().lower()

    is_aligned = predicted_class in report_text

    return {
        "predicted_class": predicted_class,
        "is_aligned": is_aligned,
        "score": 1.0 if is_aligned else 0.0,
    }


if __name__ == "__main__":
    sample_report = "The predicted price class for this product is high."
    result = check_report_alignment("high", sample_report)

    print("Report Evaluation Result:")
    print(result)