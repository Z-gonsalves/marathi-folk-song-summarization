
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT_DIR / "outputs" / "final_preprocessed_dataset.csv"
OUTPUT_DIR = ROOT_DIR / "evaluation"
OUTPUT_FILE = OUTPUT_DIR / "reference_summaries.csv"

df = pd.read_csv(INPUT_FILE, encoding="utf-8")

df = df.dropna(subset=["Title", "Lyrics"])
df = df[df["Lyrics"].astype(str).str.strip() != ""]

sample = df.sample(n=min(20, len(df)), random_state=42)

evaluation_df = sample[["Title", "Lyrics"]].copy()
evaluation_df["Reference_Summary"] = ""

OUTPUT_DIR.mkdir(exist_ok=True)
evaluation_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("Evaluation dataset created successfully!")
print(f"Songs selected: {len(evaluation_df)}")
print(f"Saved to: {OUTPUT_FILE}")