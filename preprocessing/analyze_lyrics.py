import pandas as pd
import re

# Load cleaned dataset
df = pd.read_csv("outputs/cleaned_dataset.csv", encoding="utf-8")

lyrics = " ".join(df["Lyrics"].astype(str))

print("=" * 60)
print("LYRICS ANALYSIS")
print("=" * 60)

# Total characters
print("\nTotal Characters:")
print(len(lyrics))

# Total words
words = lyrics.split()
print("\nTotal Words:")
print(len(words))

# English words
english = re.findall(r"[A-Za-z]+", lyrics)

print("\nEnglish Words Found:")
print(len(english))

# Digits
digits = re.findall(r"[0-9०-९]+", lyrics)

print("\nNumbers Found:")
print(len(digits))

# URLs
urls = re.findall(r"http\S+|www\S+", lyrics)

print("\nURLs Found:")
print(len(urls))

# Emails
emails = re.findall(r"\S+@\S+", lyrics)

print("\nEmails Found:")
print(len(emails))

# Emojis (basic range)
emoji = re.findall(r"[\U00010000-\U0010ffff]", lyrics)

print("\nEmoji Count:")
print(len(emoji))

# Punctuation count
punctuation = re.findall(r"[^\w\s]", lyrics)

print("\nPunctuation Characters:")
print(len(punctuation))