import pandas as pd

# Load dataset
df = pd.read_csv("outputs/preprocessed_dataset.csv", encoding="utf-8")

# Load stopwords
with open("preprocessing/stopwords_marathi.txt", "r", encoding="utf-8") as f:
    stopwords = set(word.strip() for word in f)

# Remove stopwords
def remove_stopwords(text):
    words = text.split()
    words = [word for word in words if word not in stopwords]
    return " ".join(words)

df["Processed_Lyrics"] = df["Processed_Lyrics"].apply(remove_stopwords)

# Save
df.to_csv(
    "outputs/final_preprocessed_dataset.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Stopwords removed successfully!")
print("Saved as outputs/final_preprocessed_dataset.csv")