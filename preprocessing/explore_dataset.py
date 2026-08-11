import pandas as pd

# Load the dataset
df = pd.read_csv("outputs/cleaned_dataset.csv", encoding="utf-8")

print("=" * 50)
print("MARATHI FOLK SONGS DATASET")
print("=" * 50)

# First 5 rows
print("\nFirst 5 Records:")
print(df.head())

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumns:")
print(df.columns.tolist())

# Data types
print("\nData Types:")
print(df.dtypes)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Basic statistics
print("\nBasic Information:")
print(df.info())

# Number of songs in each genre
print("\nGenre Distribution:")
print(df["Genre"].value_counts())

# Number of songs in each region
print("\nRegion Distribution:")
print(df["Region"].value_counts())

# Length of each lyric
df["Lyrics_Length"] = df["Lyrics"].apply(len)

print("\nAverage Lyrics Length:")
print(df["Lyrics_Length"].mean())

print("\nShortest Song:")
print(df.loc[df["Lyrics_Length"].idxmin(), ["Title", "Lyrics_Length"]])

print("\nLongest Song:")
print(df.loc[df["Lyrics_Length"].idxmax(), ["Title", "Lyrics_Length"]])