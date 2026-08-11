import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load dataset
df = pd.read_csv("outputs/final_preprocessed_dataset.csv", encoding="utf-8")

# Combine all processed lyrics
text = " ".join(df["Processed_Lyrics"].astype(str))

# Create word cloud
wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    collocations=False,
    font_path="fonts/NotoSansDevanagari-VariableFont_wdth,wght.ttf"
).generate(text)

# Show
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")

# Save
plt.savefig("visualizations/wordcloud.png", dpi=300, bbox_inches="tight")
plt.show()

print("Word cloud saved as visualizations/wordcloud.png")