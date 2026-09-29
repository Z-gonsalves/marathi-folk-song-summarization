
import re
from collections import Counter


def split_sentences(text):
    text = re.sub(r"[॥]+", "।", text)
    sentences = re.split(r"[।!?|\n]+", text)

    return [
        sentence.strip()
        for sentence in sentences
        if len(re.findall(r"\w+", sentence)) >= 4
    ]


def summarize_extractive(text, num_sentences=3):
    if not isinstance(text, str) or not text.strip():
        return ""

    sentences = split_sentences(text)

    if not sentences:
        return ""

    if len(sentences) <= num_sentences:
        return "। ".join(sentences) + "।"

    words = re.findall(r"\w+", text.lower())
    word_freq = Counter(words)

    max_freq = max(word_freq.values(), default=1)

    word_scores = {
        word: freq / max_freq
        for word, freq in word_freq.items()
    }

    sentence_scores = []

    for index, sentence in enumerate(sentences):
        sentence_words = re.findall(r"\w+", sentence.lower())

        if not sentence_words:
            continue

        frequency_score = sum(
            word_scores.get(word, 0)
            for word in sentence_words
        ) / len(sentence_words)

        length_score = min(len(sentence_words) / 10, 1)

        position_score = 1 - (index / len(sentences)) * 0.2

        total_score = (
            0.7 * frequency_score
            + 0.2 * length_score
            + 0.1 * position_score
        )

        sentence_scores.append((index, total_score, sentence))

    selected = sorted(
        sentence_scores,
        key=lambda item: item[1],
        reverse=True
    )[:num_sentences]

    selected = sorted(selected, key=lambda item: item[0])

    return "। ".join(item[2] for item in selected) + "।"


if __name__ == "__main__":
    lyrics = input("Enter Marathi folk song lyrics:\n")
    summary = summarize_extractive(lyrics)

    print("\nExtractive Summary:")
    print(summary)