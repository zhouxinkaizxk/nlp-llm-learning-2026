import pandas as pd

df = pd.read_csv("data/week03_reviews.csv")
print(df.head())
print("行数和列数：", df.shape)
print("列名：", df.columns.tolist())

label_counts = df["label"].value_counts()
print("标签分布：")
print(label_counts)

df["text_length"] = df["text"].str.len()
print(df[["text", "text_length"]].head())
print("最小长度", df["text_length"].min())
print("最大长度", df["text_length"].max())
print("平均长度", df["text_length"].mean())
print("中位数长度", df["text_length"].median())
empty_text = df["text"].fillna("").str.strip() == ""
print("空文本数量", empty_text.sum())
nonempty_text = df[~empty_text]
duplicated_text = nonempty_text.duplicated(subset=["text"])
print("所有列都重复的行数量\n",nonempty_text.duplicated().sum())
print("重复文本数量", duplicated_text.sum())
print("重复的文本\n",nonempty_text[duplicated_text]["text"].tolist())
clean_df = df[~empty_text].drop_duplicates(subset=["text", "label"])
samples = clean_df.groupby("label").sample(n=5,random_state=42)
print("每个标签的样本\n",samples)
print(samples[["text", "label"]].to_string(index=False))
longest_5 = clean_df.sort_values(by="text_length", ascending=False).head(5)
shortest_5 = clean_df.sort_values(by="text_length", ascending=True).head(5)
print("最长的5条评论\n", longest_5[["text","label"]].to_string(index=False))
print("最短的5条评论\n", shortest_5[["text","label"]].to_string(index=False))

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plot_counts = clean_df["label"].value_counts()
plt.figure(figsize=(6,4))
plt.bar(plot_counts.index, plot_counts.values)
plt.title("清洗后评论的类别分布")
plt.xlabel("情感标签")
plt.ylabel("评论数量")
plt.tight_layout()
plt.savefig("outputs/week03_label_distribution.png", dpi=300)
plt.show()

plt.hist(clean_df["text_length"], bins=10, edgecolor="black")
plt.title("清洗后评论的文本长度分布")
plt.xlabel("文本长度")
plt.ylabel("评论数量")
plt.tight_layout()
plt.savefig("outputs/week03_text_length_distribution.png", dpi=300)
plt.show()
