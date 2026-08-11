import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Load dataset
df = pd.read_csv("outputs/final_preprocessed_dataset.csv", encoding="utf-8")

# Load Marathi font
font_path = "fonts/NotoSansDevanagari-VariableFont_wdth,wght.ttf"
font_prop = font_manager.FontProperties(fname=font_path)

# Top 15 regions
region_counts = df["Region"].value_counts().head(15)

plt.figure(figsize=(12, 6))
region_counts.plot(kind="bar")

plt.title("Top 15 Regions", fontproperties=font_prop)
plt.xlabel("Region", fontproperties=font_prop)
plt.ylabel("Number of Songs", fontproperties=font_prop)

plt.xticks(rotation=45)

ax = plt.gca()

for label in ax.get_xticklabels():
    label.set_fontproperties(font_prop)

for label in ax.get_yticklabels():
    label.set_fontproperties(font_prop)

plt.tight_layout()

plt.savefig("visualizations/region_distribution.png", dpi=300)

plt.show()

print("Region distribution chart saved.")