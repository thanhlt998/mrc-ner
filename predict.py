from fastai.text.all import (
    load_learner, Learner,
)
from utils.change_query import DEFAULT_QUERY_MAPPING
import numpy as np
from datasets.mrc_ner_dataset_vlsp_fastai import SentenceTransform

path = ''
learner: Learner = load_learner(path, cpu=False)
sent_tfms = SentenceTransform(
    tokenizer=None,
    possible_only=False,
)


def predict(learner, s, tokenizer):
    labels = list(DEFAULT_QUERY_MAPPING.keys())
    data = [{
        'context': s,
        'entity_label': entity_label,
        'query': DEFAULT_QUERY_MAPPING[entity_label],
    } for entity_label in labels]
    data = [sent_tfms(sent) for sent in data]
    dl = learner.dls.test_dl(data)
    preds = learner.get_preds(dl=dl)
    start_logits, end_logits, span_logits = preds[0]
    start_preds, end_preds, match_preds = start_logits > 0, end_logits > 0, span_logits > 0
    bsz, seq_len = start_preds.size()
    match_preds = (match_preds
                   & start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                   & end_preds.unsqueeze(1).expand(-1, seq_len, -1)).numpy()
    spans = []
    for i, label in enumerate(labels):
        start_indices, end_indices = np.where(match_preds[i])
        context_si = data[i]['token_ids'].index(2) + 1
        start_indices = start_indices - context_si
        end_indices = end_indices - context_si
        span_indices = [(s_i, e_i) for s_i, e_i in zip(start_indices, end_indices) if s_i <= e_i]
        token_ids = data[i]['token_ids'][context_si:-1]
        spans.extend([
            (s_i, e_i + 1, label) for s_i, e_i in span_indices
        ])

    indxs = set()
    for s_i, e_i, _ in spans:
        indxs.add(s_i)
        indxs.add(e_i)
    indxs.add(0)
    indxs.add(len(token_ids))
    indxs = list(indxs)

    idx_map = {}
    curr_len = 0
    tokens = []
    for s_i, e_i in zip(indxs[:-1], indxs[1:]):
        text = tokenizer.decode(token_ids[s_i:e_i])
        new_tokens = text.split()
        text_len = len(new_tokens)
        tokens.extend(new_tokens)
        idx_map[s_i] = curr_len
        curr_len += text_len
    idx_map[indxs[-1]] = len(tokens)

    new_spans = [(idx_map[span[0]], idx_map[span[1]], span[2]) for span in spans]
    return new_spans, tokens
