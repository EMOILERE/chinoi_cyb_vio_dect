import pandas as pd

# 加载数据
data1 = pd.read_excel(r'E:\chinese_senti\formspring-trans.xlsx')
data2 = pd.read_excel(r'E:\chinese_senti\hinglish-trans.xlsx')
data3 = pd.read_excel(r'E:\chinese_senti\labeled_data-trans.xlsx')

# 处理数据1
def extract_from_data1(df):
    texts = []
    labels = []
    for i, row in df.iterrows():
        question = str(row['问题']) if not pd.isna(row['问题']) else ""
        answer1 = str(row['答案1']) if not pd.isna(row['答案1']) else ""
        answer2 = str(row['答案2']) if not pd.isna(row['答案2']) else ""
        answer3 = str(row['答案3']) if not pd.isna(row['答案3']) else ""
        text = question + " " + answer1 + " " + answer2 + " " + answer3
        text = text.strip()  # 移除前后多余的空白
        texts.append(text)
        labels.append(1 if row['欺负1'] == 1 or row['欺负2'] == 1 or row['欺负3'] == 1 else 0)
    return pd.DataFrame({'text': texts, 'label': labels})

data1_processed = extract_from_data1(data1)

# 处理数据2
data2_processed = data2.rename(columns={'标题': 'text', '标签': 'label'})

# 处理数据3
def extract_from_data3(df):
    texts = df['鸣叫'].tolist()
    labels = [1 if row['仇恨言论'] == 3 or row['冒犯性语言'] == 3 else 0 for i, row in df.iterrows()]
    return pd.DataFrame({'text': texts, 'label': labels})

data3_processed = extract_from_data3(data3)

# 合并数据集
all_data = pd.concat([data1_processed, data2_processed, data3_processed], ignore_index=True)

# 检查数据集
print(all_data.head())

# 保存处理后的数据
all_data.to_csv('processed_data.csv', index=False)
