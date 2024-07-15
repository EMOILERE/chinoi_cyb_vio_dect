import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np


df = pd.read_excel('E:\chinese_senti\data\processed_reviews_with_sentiment_intensity.xlsx').sample(10)
# 定义自定义数据集
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, intensities, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.intensities = intensities
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        intensity = self.intensities[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long),
            'intensity': torch.tensor(intensity, dtype=torch.float)
        }


tokenizer = BertTokenizer.from_pretrained('./local_model_cache')


# 修改模型以输出情感强度
class CustomBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.regression = torch.nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, labels=None, intensity=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)
        regression_output = self.regression(pooled_output).squeeze(-1)

        loss = None
        if labels is not None and intensity is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            classification_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            mse_loss = torch.nn.MSELoss()
            regression_loss = mse_loss(regression_output, intensity)
            loss = classification_loss + regression_loss

        # 始终返回三个值，即使在评估模式下也是如此
        return loss, logits, regression_output


model = CustomBertForSequenceClassification.from_pretrained('./local_model_cache', num_labels=2)

train_texts, val_texts, train_labels, val_labels, train_intensities, val_intensities = train_test_split(
    df['review'].tolist(), df['label'].tolist(), df['sentiment_intensity'].tolist(), test_size=0.2, random_state=42
)

train_dataset = SentimentDataset(train_texts, train_labels, train_intensities, tokenizer, max_len=128)
val_dataset = SentimentDataset(val_texts, val_labels, val_intensities, tokenizer, max_len=128)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 设置优化器
optimizer = AdamW(model.parameters(), lr=2e-5)


# 训练函数
def train(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        intensities = batch['intensity'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, intensity=intensities)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    return total_loss / len(data_loader)


# 评估函数
def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    predicted_intensities = []
    actual_intensities = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            intensities = batch['intensity'].to(device)

            _, logits, regression_output = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(logits, dim=1)

            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
            predicted_intensities.extend(regression_output.cpu().tolist())
            actual_intensities.extend(intensities.cpu().tolist())

    accuracy = accuracy_score(actual_labels, predictions)
    mse = mean_squared_error(actual_intensities, predicted_intensities)
    return accuracy, mse

num_epochs = 5
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, device)
    val_accuracy, val_mse = evaluate(model, val_loader, device)
    print(f'Epoch: {epoch + 1}')
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Validation Accuracy: {val_accuracy:.4f}')
    print(f'Validation MSE: {val_mse:.4f}')
    print()

torch.save(model.state_dict(), 'small_score_trend.pth')


def predict_sentiment_and_intensity(text):
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        _, logits, regression_output = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(logits, dim=1)
        sentiment = preds.item()
        intensity = regression_output.item()

    return sentiment, intensity


def batch_predict(texts):
    model.eval()
    results = []
    for text in texts:
        sentiment, intensity = predict_sentiment_and_intensity(text)
        results.append({
            'text': text,
            'sentiment': 'positive' if sentiment == 1 else 'negative',
            'intensity': intensity
        })
    return results


test_texts = [
    "这家餐厅的服务态度很差",
    "今天的天气真是太美好了，心情舒畅！",
    "这本书内容平平无奇，没什么特别之处。",
    "早餐很一般.房间里还算干净,就是噪音太大,价格相对来说还是贵的",
    "个头合适，口感和色泽不错，很脆，节日没有影响物流速度(○?ε?○)！",
    "外形简约，性能强劲，能够满足多数办公和影音娱乐需求，系统装的也很快，注意网卡的安装与其他驱动程序安装的方法稍有不同，需要注意。",
    "酒店的地理位置非常棒,住的高级商务间.感觉房间非常小.门童和服务生都非常热情.还有免费的水果.地毯和装修有些陈旧~",
    "很不错的酒店,可能是兰溪最好的酒店.只是位置离市中心远了点.看到了上海的旅游团也住这里.",
]


batch_results = batch_predict(test_texts)
for result in batch_results:
    print(f"文本: {result['text']}")
    print(f"情感: {result['sentiment']}")
    print(f"强度: {result['intensity']:.4f}")
    print()



