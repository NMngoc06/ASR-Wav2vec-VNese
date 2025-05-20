import sys
sys.path.append("../")
import torch
import numpy as np
from utils.feature import load_wav
from typing import Dict

class DefaultCollate:
    def __init__(self, processor, sr) -> None:
        self.processor = processor
        self.sr = sr

    def __call__(self, inputs) -> Dict[str, torch.Tensor]:
        features, transcripts = zip(*inputs)
        features = list(features)
        transcripts = list(transcripts)

        # ======= Kiểm tra audio từng file =======
        for i, feat in enumerate(features):
            if not isinstance(feat, np.ndarray):
                raise TypeError(f"[ERROR] Feature {i} is not a numpy array.")

            if feat.dtype != np.float32:
                print(f"[WARNING] Feature {i} is not float32. Found {feat.dtype}. Will cast.")
                features[i] = feat.astype(np.float32)

            if np.isnan(feat).any() or np.isinf(feat).any():
                raise ValueError(f"[ERROR] Feature {i} contains NaN or Inf.")

            duration = len(feat) / self.sr
            if duration < 0.3 or duration > 30:
                print(f"[WARNING] Audio {i} has unusual duration: {duration:.2f}s")

        # ======= Kiểm tra transcript =======
        for i, t in enumerate(transcripts):
            if not isinstance(t, str) or len(t.strip()) == 0:
                raise ValueError(f"[ERROR] Transcript {i} is empty or invalid.")

        # Xử lý phần audio (features) với feature extractor
        batch = self.processor.feature_extractor(
            features, sampling_rate=self.sr, padding=True, return_tensors="pt", return_attention_mask=True
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
    def __init__(self, data, sr, preload_data, transform=None):
        self.data = data
        self.sr = sr
        self.transform = transform
        self.preload_data = preload_data
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx) -> tuple:
        item = self.data.iloc[idx]

        # Load audio
        if not self.preload_data:
            feature = load_wav(item['path'], sr=self.sr)
        else:
            feature = item['wav']

        # Normalize kiểu dữ liệu
        if isinstance(feature, torch.Tensor):
            feature = feature.detach().cpu().numpy()

        if feature.dtype != np.float32:
            feature = feature.astype(np.float32)

        return feature, item['transcript']
