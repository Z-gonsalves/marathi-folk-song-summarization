import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("Model loaded successfully.")

df = pd.read_csv(
    "dataset/Marathi_Folk_Songs.csv",
    encoding="utf-8"
)

lyrics = str(df.loc[0, "Lyrics"])

print("\nOriginal Lyrics:\n")
print(lyrics[:1000])

print("\nTokenizing lyrics...\n")

inputs = tokenizer(
    lyrics,
    return_tensors="pt",
    truncation=True,
    max_length=512
)

print("Input tokens:", inputs["input_ids"].shape[1])

print("\nTokenized Text Preview:\n")
print(
    tokenizer.decode(
        inputs["input_ids"][0],
        skip_special_tokens=True
    )[:1000]
)

print("\nGenerating summary...\n")

with torch.no_grad():
    output = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=84,
        min_length=15,
        num_beams=4,
        no_repeat_ngram_size=2,
        do_sample=False
    )

summary = tokenizer.decode(
    output[0],
    skip_special_tokens=True
)

print("\nGenerated Summary:\n")
print(summary)