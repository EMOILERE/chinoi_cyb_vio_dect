import torch
from transformers import BertTokenizer, BertForSequenceClassification


model = BertForSequenceClassification.from_pretrained(r'E:\chinese_senti\sentiment_model_finetuned3')
tokenizer = BertTokenizer.from_pretrained(r'E:\chinese_senti\sentiment_tokenizer_finetuned3')


def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    sentiment = '正向' if predictions.item() == 1 else '负向'
    return sentiment

while 1:
    print('next:')
    user_input = input()
    result = predict_sentiment(user_input)
    print(f'情感: {result}')
