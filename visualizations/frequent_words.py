import pandas as pd
from collections import Counter

# Load dataset
df = pd.read_csv("outputs/final_preprocessed_dataset.csv", encoding="utf-8")

# Combine all processed lyrics
text = " ".join(df["Processed_Lyrics"].astype(str))

# Split into words
words = text.split()

# Count frequencies
word_counts = Counter(words)

# Top 20 words
top_20 = word_counts.most_common(20)

print("\nTop 20 Most Frequent Words:\n")
for word, count in top_20:
    print(f"{word:<20} {count}")

# Save to CSV
pd.DataFrame(top_20, columns=["Word", "Count"]).to_csv(
    "outputs/top_20_words.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved as outputs/top_20_words.csv")