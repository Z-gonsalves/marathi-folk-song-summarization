
from transformers import pipeline

MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"

summarizer = None


def load_model():
    global summarizer

    if summarizer is None:
        print("Loading summarization model...")
        summarizer = pipeline(
            "summarization",
            model=MODEL_NAME
        )

    return summarizer


def summarize_text(lyrics):
    if not isinstance(lyrics, str) or not lyrics.strip():
        return ""

    model = load_model()

    result = model(
        lyrics.strip(),
        max_length=84,
        min_length=15,
        num_beams=4,
        no_repeat_ngram_size=2,
        do_sample=False,
        truncation=True
    )

    return result[0]["summary_text"].strip()