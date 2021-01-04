from utils.read_file import load_jsonl
from fastai.text.all import (
    Transform, AttrGetter, ItemGetter,
    Categorize, CategoryMap,
    Datasets,
    pad_input, partial, noop,
)
from transformers import PhobertTokenizer
import numpy as np
import torch
from typing import List


class SentenceTransform(Transform):
    def __init__(
            self,
            tokenizer: PhobertTokenizer,
            possible_only=False,
    ):
        super(SentenceTransform, self).__init__()
        self.tokenizer = tokenizer
        self.possible_only = possible_only

    def encodes(self, x: dict):
        tokenizer = self.tokenizer
        query = x['query']
        context = x['context']
        start_positions = x.get("start_position")
        end_positions = x.get("end_position")

        context_words = context.split()
        context_sub_tokens = [tokenizer.encode(word, add_special_tokens=False) for word in context_words]

        query_words = query.split()
        query_sub_tokens = [tokenizer.encode(word, add_special_tokens=False) for word in query_words]

        tokens = [
            tokenizer.cls_token_id,
            *(sum(query_sub_tokens, [])), tokenizer.sep_token_id,
            *(sum(context_sub_tokens, [])), tokenizer.sep_token_id,
        ]

        res = {
            'token_ids': tokens,
            **{
                k: v for k, v in x.items() if k not in ['query', 'context']
            }
        }

        if start_positions is not None and end_positions is not None:
            context_word_lengths = [len(sub_tokens) for sub_tokens in context_sub_tokens]
            query_word_lengths = [len(sub_tokens) for sub_tokens in query_sub_tokens]
            query_length = sum(query_word_lengths)
            context_word_lengths_cumsum = np.cumsum([query_length + 2, *context_word_lengths])
            start_positions = [context_word_lengths_cumsum[p] for p in start_positions]
            end_positions = [context_word_lengths_cumsum[p + 1] - 1 for p in end_positions]

            res['start_position'] = start_positions
            res['end_position'] = end_positions
            res['context_word_lengths_cumsum'] = context_word_lengths_cumsum

        return res

    def decodes(self, x):
        token_ids = x['token_ids']
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        return {
            'token_str': ' '.join(tokens),
            'start_position': x['start_position'],
            'end_position': x['end_position'],
            'str': self.tokenizer.convert_tokens_to_string(tokens),
        }


class YTransform(Transform):
    def __init__(self):
        super(YTransform, self).__init__()

    def encodes(self, x):
        token_ids = x['token_ids']
        context_word_lengths_cumsum = x['context_word_lengths_cumsum']
        start_positions = x['start_position']
        end_positions = x['end_position']

        # start, end label mask
        start_label_mask = np.zeros(len(token_ids))
        end_label_mask = np.zeros(len(token_ids))

        start_label_mask[context_word_lengths_cumsum[:-1]] = 1
        end_label_mask[context_word_lengths_cumsum[1:] - 1] = 1

        # start, end labels
        start_labels = np.zeros(len(token_ids))
        end_labels = np.zeros(len(token_ids))

        start_labels[start_positions] = 1
        end_labels[end_positions] = 1

        # match labels
        seq_len = len(token_ids)
        match_labels = torch.zeros([seq_len, seq_len], dtype=torch.long)
        match_labels[start_positions, end_positions] = 1

        return (
            torch.LongTensor(start_labels),
            torch.LongTensor(end_labels),
            torch.LongTensor(start_label_mask),
            torch.LongTensor(end_label_mask),
            match_labels,
        )


class BeforeBatchTransform(Transform):
    def __init__(
            self,
            max_seq_length,
            pad_fields: List[int],
            pad_values: List[int],
            pad_token_id: int,
            sep_token_id: int,
    ):
        super(BeforeBatchTransform, self).__init__()
        assert len(pad_fields) == len(pad_values), 'no.fields must be equal to no.values'
        self.max_seq_length = max_seq_length
        self.pad_fields = pad_fields
        self.pad_values = pad_values
        self.pad_token_id = pad_token_id
        self.sep_token_id = sep_token_id

        self.pad_funcs = {
            pad_field: partial(pad_input, pad_idx=pad_value, pad_fields=pad_field, pad_first=False,)
            for pad_value, pad_field in zip(pad_values, pad_fields)
        }

        self.pad_funcs[5] = self._pad_match_labels

    def encodes(self, items):
        items = [self._truncate(item) for item in items]
        n_fields = len(items[0])
        for i in range(n_fields):
            items = self.pad_funcs.get(i, noop)(items)

        return items

    def _pad_match_labels(self, items):
        if len(items[0]) < 6: return items
        new_items = []
        max_len = items[0][0].size(0)
        for *item, match_labels in items:
            pad_match_labels = torch.zeros((max_len, max_len), dtype=torch.long)
            pad_match_labels[:match_labels.size(0), :match_labels.size(1)] = match_labels
            new_items.append((*item, pad_match_labels))
        return new_items

    def _truncate(self, item: tuple):
        if len(item) == 2:
            item = (item[0], *item[1])
        token_ids = item[0]
        seq_len = len(token_ids)
        if seq_len <= self.max_seq_length:
            return item

        token_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels = item

        token_ids = token_ids[:self.max_seq_length]
        start_labels = start_labels[:self.max_seq_length]
        end_labels = end_labels[:self.max_seq_length]
        start_label_mask = start_label_mask[:self.max_seq_length]
        end_label_mask = end_label_mask[:self.max_seq_length]
        match_labels = match_labels[:self.max_seq_length, :self.max_seq_length]

        token_ids[-1] = self.sep_token_id
        start_labels[-1] = 0
        end_labels[-1] = 0
        start_label_mask[-1] = 0
        end_label_mask[-1] = 0
        match_labels[-1, :] = 0
        match_labels[:, -1] = 0
        return (
            token_ids,
            start_labels,
            end_labels,
            start_label_mask,
            end_label_mask,
            match_labels,
        )


