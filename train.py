import pandas as pd
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import Dataset
import re
import os


class RumourDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        # 基本文本清洗（保留URL和话题标签）
        cleaned_texts = [re.sub(r'[\x00-\x1F\x7F]', '', str(t)) for t in texts]
        self.encodings = tokenizer(
            cleaned_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = labels.tolist()
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': torch.tensor(self.labels[idx])
        }
    
    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {'accuracy': acc, 'f1': f1}

# 加载数据
train_df = pd.read_csv('./data/train.csv')
val_df = pd.read_csv('./data/val.csv')


# 初始化tokenizer和模型
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2
)

# 准备数据集
train_dataset = RumourDataset(train_df['text'], train_df['label'], tokenizer)
val_dataset = RumourDataset(val_df['text'], val_df['label'], tokenizer)

# 训练配置 - 使用兼容性参数名
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=128,
    learning_rate=6e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    eval_strategy='steps' , #查阅资料中显示高版本的transformers中该参数为evaluation_strategy，但此处如使用该参数名则会报错
    eval_steps=200,
    save_strategy='steps',
    save_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    fp16=torch.cuda.is_available(),
    report_to='none'
)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 训练模型
trainer.train()

# 保存最佳模型
model.save_pretrained('rumour_model')
tokenizer.save_pretrained('rumour_model')
print("模型已保存到 rumour_model 目录")

# 验证集最终评估
val_results = trainer.predict(val_dataset)
val_preds = np.argmax(val_results.predictions, axis=-1)

print("\n验证集最终评估结果:")
print(classification_report(val_df['label'], val_preds))
print(f"准确率: {accuracy_score(val_df['label'], val_preds):.4f}")
print(f"F1分数: {f1_score(val_df['label'], val_preds):.4f}")

