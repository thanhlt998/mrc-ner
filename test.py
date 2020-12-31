from transformers import PhobertTokenizer
# from tokenizers import BertWordPieceTokenizer

# tokenizer = ByteLevelBPETokenizer.from_file('data/phobert/vocab.txt', 'data/phobert/merges.txt')

# tokenizer = PhobertTokenizer.from_pretrained('vinai/phobert-base')
# context = "Tôi là sinh_viên trường đại_học Công_nghệ ."
# query = "tìm trường đại_học trong văn_bản"
#
# res = tokenizer.encode(query, context)
# print(res, type(res))
# print(tokenizer.decode(res))


# tokenizer = BertWordPieceTokenizer.from_file('data/bert-base-uncased/vocab.txt')
# context = "There are many documents on the table"
# query = "which is on the table"
#
# res = tokenizer.encode(query, context)
# print('words', res.words)
# print('tokens', res.tokens)
# print('ids', res.ids)
# print('type_ids', res.type_ids)
# print('offsets', res.offsets)

# from datasets.mrc_ner_dataset_vlsp2 import run_dataset
from datasets.mrc_ner_dataset_vlsp import run_dataset
run_dataset()
