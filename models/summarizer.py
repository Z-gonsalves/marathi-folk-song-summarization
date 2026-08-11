import pandas as pd
from transformers import pipeline

print("Loading summarization model...")

# Load multilingual summarization model
summarizer = pipeline(
    "summarization",
    model="csebuetnlp/mT5_multilingual_XLSum"
)

# Load dataset
df = pd.read_csv("outputs/final_preprocessed_dataset.csv", encoding="utf-8")

# Take one song for testing
lyrics = df.loc[0, "Processed_Lyrics"]

print("\nOriginal Lyrics:\n")
print(lyrics[:1000])   # Show first 1000 characters

print("\nGenerating summary...\n")

summary = summarizer(
    lyrics,
    max_length=80,
    min_length=25,
    do_sample=False
)

print("Summary:\n")
print(summary[0]["summary_text"])