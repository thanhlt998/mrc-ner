import torch
from tokenizers import BertWordPieceTokenizer
from transformers import PhobertTokenizer
from torch.utils.data import Dataset
from utils.read_file import load_jsonl
import numpy as np


class MRCNERDataset(Dataset):
    """
    MRC NER Dataset
    Args:
        json_path: path to mrc-ner style json
        tokenizer: BertTokenizer
        max_length: int, max length of query+context
        possible_only: if True, only use possible samples that contain answer for the query/context
    """

    def __init__(self, jsonl_path, tokenizer: PhobertTokenizer, max_length: int = 256, possible_only=False,
                 pad_to_maxlen=False):
        self.all_data = load_jsonl(jsonl_path)
        self.tokenzier = tokenizer
        self.max_length = max_length
        self.possible_only = possible_only
        if self.possible_only:
            self.all_data = [
                x for x in self.all_data if x["start_position"]
            ]
        self.pad_to_maxlen = pad_to_maxlen

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item: int):
        """
        Args:
            item: int, idx
        Returns:
            tokens: tokens of query + context, [seq_len]
            token_type_ids: token type ids, 0 for query, 1 for context, [seq_len]
            start_labels: start labels of NER in tokens, [seq_len]
            end_labels: end labels of NER in tokens, [seq_len]
            label_mask: label mask, 1 for counting into loss, 0 for ignoring. [seq_len]
            match_labels: match labels, [seq_len, seq_len]
            sample_idx: sample id
            label_idx: label id

        """
        data = self.all_data[item]
        tokenizer = self.tokenzier

        qas_id = data.get("qas_id", "0.0")
        sample_idx, label_idx = qas_id.split(".")
        sample_idx = torch.LongTensor([int(sample_idx)])
        label_idx = torch.LongTensor([int(label_idx)])

        query = data["query"]
        context = data["context"]
        start_positions = data["start_position"]
        end_positions = data["end_position"]

        # add space offsets
        context_words = context.split()
        context_sub_tokens = [tokenizer.encode(word, add_special_tokens=False) for word in context_words]
        context_word_lengths = [len(sub_tokens) for sub_tokens in context_sub_tokens]

        query_words = query.split()
        query_sub_tokens = [tokenizer.encode(word, add_special_tokens=False) for word in query_words]
        query_word_lengths = [len(sub_tokens) for sub_tokens in query_sub_tokens]

        tokens = [
            tokenizer.cls_token_id,
            *(sum(query_sub_tokens, [])), tokenizer.sep_token_id,
            *(sum(context_sub_tokens, [])), tokenizer.sep_token_id,
        ]

        query_length = sum(query_word_lengths)
        context_word_lengths_cumsum = np.cumsum([query_length + 2, *context_word_lengths])
        new_start_positions = [context_word_lengths_cumsum[p] for p in start_positions]
        new_end_positions = [context_word_lengths_cumsum[p + 1] - 1 for p in end_positions]

        type_ids = np.zeros(len(tokens))

        label_mask = type_ids.copy()
        start_label_mask = np.zeros(len(tokens))
        end_label_mask = np.zeros(len(tokens))

        start_label_mask[context_word_lengths_cumsum[:-1]] = 1
        end_label_mask[context_word_lengths_cumsum[1:] - 1] = 1

        # the start/end position must be whole word
        assert all(start_label_mask[p] != 0 for p in new_start_positions)
        assert all(end_label_mask[p] != 0 for p in new_end_positions)

        assert len(new_start_positions) == len(new_end_positions) == len(start_positions)
        assert len(label_mask) == len(tokens)
        start_labels = np.zeros(len(tokens))
        end_labels = np.zeros(len(tokens))

        start_labels[new_start_positions] = 1
        end_labels[new_end_positions] = 1

        # truncate
        tokens = tokens[: self.max_length]
        type_ids = type_ids[: self.max_length]
        start_labels = start_labels[: self.max_length]
        end_labels = end_labels[: self.max_length]
        start_label_mask = start_label_mask[: self.max_length]
        end_label_mask = end_label_mask[: self.max_length]

        # make sure last token is [SEP]
        sep_token = tokenizer.sep_token_id
        if tokens[-1] != sep_token:
            assert len(tokens) == self.max_length
            tokens[-1] = sep_token
            start_labels[-1] = 0
            end_labels[-1] = 0
            start_label_mask[-1] = 0
            end_label_mask[-1] = 0

        if self.pad_to_maxlen:
            tokens = self.pad(tokens, tokenizer.pad_token_id)
            type_ids = self.pad(type_ids, 0)
            start_labels = self.pad(start_labels)
            end_labels = self.pad(end_labels)
            start_label_mask = self.pad(start_label_mask)
            end_label_mask = self.pad(end_label_mask)

        seq_len = len(tokens)
        match_labels = torch.zeros([seq_len, seq_len], dtype=torch.long)
        for start, end in zip(new_start_positions, new_end_positions):
            if start >= seq_len or end >= seq_len:
                continue
            match_labels[start, end] = 1

        return [
            torch.LongTensor(tokens),
            torch.LongTensor(type_ids),
            torch.LongTensor(start_labels),
            torch.LongTensor(end_labels),
            torch.LongTensor(start_label_mask),
            torch.LongTensor(end_label_mask),
            match_labels,
            sample_idx,
            label_idx
        ]

    def pad(self, lst, value=0, max_length=None):
        max_length = max_length or self.max_length
        if len(lst) < max_length:
            res = np.full(max_length, fill_value=value, dtype=int)
            res[:len(lst)] = lst
        return lst


def run_dataset():
    """test dataset"""
    import os
    from datasets.collate_functions_vlsp import collate_to_max_length
    from torch.utils.data import DataLoader
    # zh datasets
    # bert_path = "/mnt/mrc/chinese_L-12_H-768_A-12"
    # json_path = "/mnt/mrc/zh_msra/mrc-ner.test"
    # # json_path = "/mnt/mrc/zh_onto4/mrc-ner.train"
    # is_chinese = True

    # en datasets
    jsonl_path = "data/vlsp_processed/train.jsonl"

    tokenizer = PhobertTokenizer.from_pretrained('vinai/phobert-base')
    dataset = MRCNERDataset(jsonl_path=jsonl_path, tokenizer=tokenizer, pad_to_maxlen=False, max_length=60)

    dataloader = DataLoader(dataset, batch_size=100,
                            collate_fn=collate_to_max_length)

    for batch in dataloader:
        for tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx in zip(
                *batch):
            tokens = tokens.tolist()
            start_positions, end_positions = torch.where(match_labels > 0)
            start_positions = start_positions.tolist()
            end_positions = end_positions.tolist()
            if not start_positions:
                continue
            print("=" * 20)
            print(f"len: {len(tokens)}", tokenizer.decode(tokens, skip_special_tokens=False))
            for start, end in zip(start_positions, end_positions):
                print(str(sample_idx.item()), str(label_idx.item()) + "\t" + tokenizer.decode(tokens[start: end + 1]))

            print('tokens', tokens)
            print(list(zip(tokenizer.convert_ids_to_tokens(tokens), start_labels.tolist())))
            print('token_type_ids', token_type_ids)
            print('start_labels', start_labels)
            print('end_labels', end_labels)
            print('start_label_mask', start_label_mask)
            print('end_label_mask', end_label_mask)
            print('match_labels', match_labels)
            print('sample_idx', sample_idx)
            print('label_idx', label_idx)
            break

        break


if __name__ == '__main__':
    run_dataset()
