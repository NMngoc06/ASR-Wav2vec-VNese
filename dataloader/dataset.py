import sys
sys.path.append("../")
import torch

from utils.feature import load_wav
from typing import Dict

class DefaultCollate:
    def __init__(self, processor, sr) -> None:
        self.processor = processor
        self.sr = sr
    def __call__(self, inputs) -> Dict[str, torch.tensor]:
        features, transcripts = zip(*inputs)
        features = list(features)
        transcripts = list(transcripts)

        # Xử lý phần audio (features) với feature extractor
        batch = self.processor.feature_extractor(
            features, sampling_rate=16000, padding=True, return_tensors="pt", return_attention_mask=True
        )

        # Xử lý phần text (transcripts) với tokenizer
        with self.processor.as_target_processor():
            labels_batch = self.processor.tokenizer(
                transcripts, padding=True, return_tensors="pt"
            )

        # Tạo labels, thay thế phần padding bằng -100 để ignore khi tính loss
        batch["labels"] = labels_batch.input_ids.masked_fill(labels_batch.attention_mask.ne(1), -100)

        return batch



class Dataset:
    def __init__(self, data, sr, preload_data, transform = None):
        self.data = data
        self.sr = sr
        self.transform = transform
        self.preload_data = preload_data
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx) -> tuple:
        item = self.data.iloc[idx]
        if not self.preload_data:
            feature = load_wav(item['path'], sr = self.sr)
        else:
            feature = item['wav']
        
        return feature, item['transcript']

