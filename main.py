from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(Path(__file__).with_name("student_performance.csv"))
print("Preview:\n", df.head())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nDescriptive statistics:\n", df.describe())
scores = df["exam_score"].to_numpy()
print("\nMean:", np.mean(scores))
print("Median:", np.median(scores))
print("Standard deviation:", np.std(scores))
df["performance_band"] = pd.cut(df["exam_score"], [0,50,70,100],
    labels=["Needs support","Developing","Strong"], include_lowest=True)
print("\nPerformance bands:\n", df["performance_band"].value_counts().sort_index())
print("\nCorrelation (study hours and score):",
      df["hours_studied"].corr(df["exam_score"]))
plt.figure(figsize=(8,5))
plt.scatter(df["hours_studied"], df["exam_score"])
plt.xlabel("Hours studied"); plt.ylabel("Exam score")
plt.title("Study Hours vs Exam Score — Synthetic Sample")
plt.grid(alpha=.25); plt.tight_layout()
plt.savefig(Path(__file__).with_name("study_hours_vs_score.png"), dpi=160)
print("Chart saved: study_hours_vs_score.png")
