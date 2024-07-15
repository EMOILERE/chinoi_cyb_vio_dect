import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random


def predict(text):
    model = BertForSequenceClassification.from_pretrained('./final_model_1')
    tokenizer = BertTokenizer.from_pretrained('./final_model_1')
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()

    return prediction

if __name__ == "__main__":


    file_path = 'E:\chinese_senti\Further_Cleaned_Hinglish_Data.xlsx'
    df = pd.read_excel(file_path)

    random.seed(42)
    df_sample = df.sample(n=1000)

    texts = df_sample['标题'].tolist()
    true_labels = df_sample['标签'].tolist()

    predicted_labels = []
    for text in texts:
        result = predict(text)
        predicted_labels.append(result)

    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_labels)
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    print("ROC曲线绘制完成")
