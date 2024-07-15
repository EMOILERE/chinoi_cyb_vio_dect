import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import roc_curve, auc, classification_report
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

data = pd.read_excel(r'E:\chinese_senti\cleaned.xlsx').sample(1000)
data = data[['review', 'label']]

data['label'] = data['label'].astype(int)
data.dropna(subset=['review'], inplace=True)
data['review'] = data['review'].astype(str)

test_texts = data['review']
test_labels = data['label']


tokenizer = BertTokenizer.from_pretrained(r'E:\\chinese_senti\\sentiment_tokenizer_finetuned3')
model = BertForSequenceClassification.from_pretrained(r'E:\\chinese_senti\\sentiment_model_finetuned3')

def tokenize_function(texts):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=128, return_tensors='pt')

test_encodings = tokenize_function(test_texts.tolist())

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

test_dataset = SentimentDataset(test_encodings, test_labels.tolist())
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_loader:
        inputs = {key: val for key, val in batch.items() if key != 'labels'}
        labels = batch['labels']
        outputs = model(**inputs)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=-1).numpy())
        true_labels.extend(labels.numpy())

fpr, tpr, _ = roc_curve(true_labels, predictions)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')

writer = SummaryWriter('runs2/sentiment_analysis')

dummy_input = torch.zeros(1, 128, dtype=torch.long)
attention_mask = torch.ones(1, 128, dtype=torch.long)
inputs = (dummy_input, attention_mask)
traced_model = torch.jit.trace(model, inputs, strict=False)
writer.add_graph(traced_model, inputs)


fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic')
ax.legend(loc="lower right")

writer.add_figure('ROC curve', fig)
writer.close()


report = classification_report(true_labels, predictions, target_names=['negative', 'positive'])
print('Classification Report:')
print(report)
