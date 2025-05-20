from evaluate import load
import torch

class Metric:
    def __init__(self, processor):
        self.processor = processor
        self.wer_metric = load("wer")  # sửa load_metric thành load từ evaluate

    def __call__(self, logits, labels):
        preds = torch.argmax(logits, axis=-1)

        # Thay giá trị -100 (padding mask) bằng pad_token_id để decode đúng
        labels = labels.clone()  # tránh sửa trực tiếp tensor input
        labels[labels == -100] = self.processor.tokenizer.pad_token_id

        pred_strs = self.processor.batch_decode(preds)
        label_strs = self.processor.batch_decode(labels, group_tokens=False)

        wer = self.wer_metric.compute(predictions=pred_strs, references=label_strs)
        return wer
