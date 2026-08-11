import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Load font from project folder
font_path = "fonts/NotoSansDevanagari-VariableFont_wdth,wght.ttf"
font_prop = font_manager.FontProperties(fname=font_path)

# Load dataset
df = pd.read_csv("outputs/final_preprocessed_dataset.csv", encoding="utf-8")

# Count genres
genre_counts = df["Genre"].value_counts()

import matplotlib

matplotlib.rcParams["font.family"] = "Noto Sans Devanagari"

# Plot
plt.figure(figsize=(10, 6))
genre_counts.plot(kind="bar")

plt.title("Genre Distribution", fontproperties=font_prop)
plt.xlabel("Genre", fontproperties=font_prop)
plt.ylabel("Number of Songs", fontproperties=font_prop)

plt.xticks(rotation=45)
ax = plt.gca()

for label in ax.get_xticklabels():
    label.set_fontproperties(font_prop)

for label in ax.get_yticklabels():
    label.set_fontproperties(font_prop)
plt.tight_layout()

# Save
plt.savefig("visualizations/genre_distribution.png", dpi=300)

plt.show()

print("Genre distribution chart saved.")