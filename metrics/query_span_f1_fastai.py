from fastai.text.all import (
    Metric,
)
from .functional.query_span_f1 import query_span_f1
import torch


class QuerySpanF1(Metric):
    def __init__(self, flat=False):
        self.flat = flat
        self.stats = []

    def reset(self):
        self.stats = []

    def accumulate(self, learn):
        start_logits, end_logits, span_logits = learn.pred
        start_labels, end_labels, start_label_mask, end_label_mask, match_labels = learn.yb
        start_preds, end_preds = start_logits > 0, end_logits > 0
        span_f1_stats = query_span_f1(
            start_preds=start_preds,
            end_preds=end_preds,
            match_logits=span_logits,
            start_label_mask=start_label_mask,
            end_label_mask=end_label_mask,
            match_labels=match_labels,
        )
        self.stats.append(span_f1_stats)

    @property
    def value(self):
        all_counts = torch.stack(self.stats).sum(0)
        span_tp, span_fp, span_fn = all_counts
        span_recall = span_tp / (span_tp + span_fn + 1e-10)
        span_precision = span_tp / (span_tp + span_fp + 1e-10)
        span_f1 = span_precision * span_recall * 2 / (span_recall + span_precision + 1e-10)
        return span_f1

    @property
    def name(self):
        return "query_span_f1"
