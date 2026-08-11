import pandas as pd
import re
import unicodedata

# Load cleaned dataset
df = pd.read_csv("outputs/cleaned_dataset.csv", encoding="utf-8")

# -----------------------------
# Function to preprocess lyrics
# -----------------------------
def preprocess(text):

    if pd.isna(text):
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove Emails
    text = re.sub(r"\S+@\S+", "", text)

    # Remove English words
    text = re.sub(r"[A-Za-z]+", "", text)

    # Remove digits (English + Marathi)
    text = re.sub(r"[0-9०-९]+", "", text)

    # Remove emojis
    text = re.sub(r"[\U00010000-\U0010ffff]", "", text)

    # Remove punctuation
    # Keep Marathi sentence markers । and ॥
    text = re.sub(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]", "", text)

    # Remove Marathi sentence markers
    text = re.sub(r"[।॥]+", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Apply preprocessing
df["Processed_Lyrics"] = df["Lyrics"].apply(preprocess)

# Save
df.to_csv(
    "outputs/preprocessed_dataset.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Preprocessing completed successfully!")
print("Saved as outputs/preprocessed_dataset.csv")