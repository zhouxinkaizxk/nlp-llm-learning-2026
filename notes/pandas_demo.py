import pandas as pd

df = pd.DataFrame({
    "text": ["很好看", None, "", "   ", "不好看", "很好看"],
    "label": ["正面", "负面", "正面", "负面", "负面", "正面"]
})

print(df)
# 判断 text 列哪些位置缺失
print(df["text"].isna())

# 统计每一列有几个缺失值
print(df.isna().sum())

# 找出 text 缺失的那些行
print(df[df["text"].isna()])

empty_text = df["text"].fillna("").str.strip() == ""

print(empty_text.sum())  # 3
print(df[empty_text])    # 找到索引 1、2、3

print(df.duplicated())

# 统计多出来的重复行数量
print(df.duplicated().sum())  # 1

# 找出这些重复行
print(df[df.duplicated()])    # 索引 5

repeated_text = df.duplicated(subset=["text"], keep=False)

print(df[repeated_text])  # 索引 0 和 5

# 去掉空文本；相同的“文本＋标签”保留第一条
clean_df = df[~empty_text].drop_duplicates(subset=["text", "label"])

print(clean_df)

clean_df.loc[4, "label"] = clean_df.loc[4, "label"].replace("负面", "正面")
clean_df = clean_df.reset_index(drop=True)
print(clean_df)
