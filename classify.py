import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import re

class RumourDetectClass:
    def __init__(self, model_path='rumour_model'):
        # 设备配置 (优先GPU 0号卡)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 加载tokenizer和模型
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        print(f"模型加载完成，使用设备: {self.device}")

    def _clean_text(self, text):
        #基本文本清洗（保留URL和话题标签）
        return re.sub(r'[\x00-\x1F\x7F]', '', text)

    def classify(self, text: str) -> int:
        #谣言分类接口
        # 清理文本
        cleaned_text = self._clean_text(text)
        
        # Tokenize
        inputs = self.tokenizer(
            cleaned_text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 解析结果
        logits = outputs.logits
        return torch.argmax(logits, dim=-1).item()