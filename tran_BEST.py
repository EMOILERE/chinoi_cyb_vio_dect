


import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


data = pd.read_excel(r"E:\chinese_senti\Further_Cleaned_Data.xlsx")
label_1 = data[data['标签'] == 1]
label_0 = data[data['标签'] == 0]

selected_label_1 = label_1.sample(n=1000, random_state=42)
selected_label_0 = label_0.sample(n=1000, random_state=42)

combined_df = pd.concat([selected_label_1, selected_label_0], ignore_index=True)

combined_df = combined_df.sample(frac=1).reset_index(drop=True)

# 加载本地BERT的中文分词器
tokenizer = BertTokenizer.from_pretrained('E:/chinese_senti/Classification_Regression/local_model_cache')

def tokenize_function(texts):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=256, return_tensors='pt')

tokenized_texts = tokenize_function(combined_df['标题'].astype(str).tolist())

# 创建Dataset
class ViolenceDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_texts, val_texts, train_labels, val_labels = train_test_split(
    tokenized_texts['input_ids'], combined_df['标签'].tolist(), test_size=0.2, random_state=42
)

train_masks, val_masks = train_test_split(tokenized_texts['attention_mask'], test_size=0.2, random_state=42)

train_dataset = ViolenceDataset({'input_ids': train_texts, 'attention_mask': train_masks}, train_labels)
val_dataset = ViolenceDataset({'input_ids': val_texts, 'attention_mask': val_masks}, val_labels)

model = BertForSequenceClassification.from_pretrained('E:/chinese_senti/Classification_Regression/local_model_cache', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results_2',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs_2',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_steps=1000,  # 每1000步保存一次模型
    save_total_limit=3,  # 最多保存3个模型
)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)


trainer.train()
results = trainer.evaluate()
print(results)

# 保存最终的模型
model.save_pretrained('./final_model_1')
tokenizer.save_pretrained('./final_model_1')
