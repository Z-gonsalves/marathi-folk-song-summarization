import pandas as pd
from collections import Counter

# Load dataset
df = pd.read_csv("outputs/final_preprocessed_dataset.csv", encoding="utf-8")

genres = [
    "पोवाडा",
    "अभंग",
    "भजन",
    "लावणी",
    "ओवी",
    "भारुड",
    "कोळीगीत"
]

for genre in genres:
    print("\n" + "=" * 60)
    print(f"Genre: {genre}")
    print("=" * 60)

    songs = df[df["Genre"] == genre]

    text = " ".join(songs["Processed_Lyrics"].astype(str))

    words = text.split()

    counter = Counter(words)

    for word, count in counter.most_common(40):
        if len(word) > 2:
            print(f"{word:<20} {count}")