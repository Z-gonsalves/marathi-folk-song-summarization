
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from rouge_score import rouge_scorer
from rouge_score import tokenizers

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from models.summarizer import summarize_text
from baseline.extractive_summarizer import summarize_extractive

INPUT_FILE = ROOT_DIR / "evaluation" / "reference_summaries.csv"
RESULT_FILE = ROOT_DIR / "evaluation" / "evaluation_results.csv"
SCORES_FILE = ROOT_DIR / "evaluation" / "rouge_scores.csv"


class MarathiTokenizer(tokenizers.Tokenizer):
    def tokenize(self, text):
        text = unicodedata.normalize("NFC", str(text)).lower()
        words = []
        current = ""

        for char in text:
            if char.isalnum():
                current += char
            elif current:
                words.append(current)
                current = ""

        if current:
            words.append(current)

        return words


def calculate_scores(scorer, reference, generated):
    scores = scorer.score(reference, generated)

    return {
        "ROUGE-1": scores["rouge1"].fmeasure,
        "ROUGE-2": scores["rouge2"].fmeasure,
        "ROUGE-L": scores["rougeL"].fmeasure
    }


def main():
    print("Loading evaluation dataset...")
    df = pd.read_csv(INPUT_FILE, encoding="utf-8")

    required_columns = ["Title", "Lyrics", "Reference_Summary"]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        tokenizer=MarathiTokenizer()
    )

    results = []

    for index, row in df.iterrows():
        title = str(row["Title"])
        lyrics = str(row["Lyrics"])
        reference = str(row["Reference_Summary"])

        if not lyrics.strip() or not reference.strip():
            print(f"Skipping row {index + 1}: empty lyrics or reference")
            continue

        print(f"\nEvaluating {index + 1}/{len(df)}: {title}")

        print("Generating extractive summary...")
        extractive_summary = summarize_extractive(lyrics)

        print("Generating mT5 summary...")
        mt5_summary = summarize_text(lyrics)

        if not isinstance(extractive_summary, str):
            extractive_summary = str(extractive_summary)

        if not isinstance(mt5_summary, str):
            mt5_summary = str(mt5_summary)

        extractive_scores = calculate_scores(
            scorer, reference, extractive_summary
        )

        mt5_scores = calculate_scores(
            scorer, reference, mt5_summary
        )

        results.append({
            "Title": title,
            "Reference_Summary": reference,
            "Extractive_Summary": extractive_summary,
            "Extractive_ROUGE-1": extractive_scores["ROUGE-1"],
            "Extractive_ROUGE-2": extractive_scores["ROUGE-2"],
            "Extractive_ROUGE-L": extractive_scores["ROUGE-L"],
            "mT5_Summary": mt5_summary,
            "mT5_ROUGE-1": mt5_scores["ROUGE-1"],
            "mT5_ROUGE-2": mt5_scores["ROUGE-2"],
            "mT5_ROUGE-L": mt5_scores["ROUGE-L"]
        })

        print("\nExtractive Summary:", extractive_summary)
        print("Extractive ROUGE:", {
            key: round(value, 4)
            for key, value in extractive_scores.items()
        })

        print("\nmT5 Summary:", mt5_summary)
        print("mT5 ROUGE:", {
            key: round(value, 4)
            for key, value in mt5_scores.items()
        })

    if not results:
        raise ValueError("No evaluation results were generated.")

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        RESULT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    metric_names = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]

    average_rows = []

    for method, prefix in [
        ("Extractive", "Extractive_"),
        ("mT5", "mT5_")
    ]:
        for metric in metric_names:
            average_rows.append({
                "Method": method,
                "Metric": metric,
                "Average F1 Score": results_df[
                    f"{prefix}{metric}"
                ].mean()
            })

    average_df = pd.DataFrame(average_rows)
    average_df.to_csv(
        SCORES_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    comparison = average_df.pivot(
        index="Metric",
        columns="Method",
        values="Average F1 Score"
    ).reset_index()

    print("\nEvaluation completed.")
    print("\nAverage ROUGE F1 Score Comparison:")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nDetailed results saved to: {RESULT_FILE}")
    print(f"Average scores saved to: {SCORES_FILE}")


if __name__ == "__main__":
    main()