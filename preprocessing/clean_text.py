import pandas as pd
import re

# Load dataset
df = pd.read_csv("dataset/Marathi_Folk_Songs.csv", encoding="utf-8")

# -----------------------------
# Clean Genre column
# -----------------------------
df["Genre"] = (
    df["Genre"]
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# -----------------------------
# Clean Region column
# -----------------------------
df["Region"] = (
    df["Region"]
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# -----------------------------
# Clean Title column
# -----------------------------
df["Title"] = (
    df["Title"]
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# -----------------------------
# Clean Lyrics column
# -----------------------------
df["Lyrics"] = (
    df["Lyrics"]
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# -----------------------------
# Clean History column
# -----------------------------
df["History"] = (
    df["History"]
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# Save cleaned dataset
df.to_csv(
    "outputs/cleaned_dataset.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Dataset cleaned successfully!")
print("Saved as outputs/cleaned_dataset.csv")