import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 加载处理过的数据
file_path = 'E:\chinese_senti\Processed_Text_and_Labels.csv'
processed_df = pd.read_csv(file_path).sample(100)

# 将数据分为训练集和测试集
_, test_df = train_test_split(processed_df, test_size=0.2, random_state=42)

model = BertForSequenceClassification.from_pretrained('./final_model')
tokenizer = BertTokenizer.from_pretrained('./final_model')

# 分词数据
def tokenize(text):
    return tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")

# 对测试集进行预测
test_texts = list(test_df['text'].astype(str))
test_labels = test_df['label'].values

all_logits = []
model.eval()
with torch.no_grad():
    for text in test_texts:
        inputs = tokenize(text)
        outputs = model(**inputs)
        logits = outputs.logits
        all_logits.append(logits.squeeze().numpy())

all_logits = torch.tensor(all_logits)
pred_probs = torch.softmax(all_logits, dim=1)[:, 1].numpy()

# 绘制ROC曲线
fpr, tpr, _ = roc_curve(test_labels, pred_probs)
roc_auc = auc(fpr, tpr)

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
