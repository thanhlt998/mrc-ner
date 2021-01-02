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
# from datasets.mrc_ner_dataset_vlsp import run_dataset
# from datasets.mrc_ner_dataset import run_dataset
# run_dataset()


# from transformers.models.roberta import RobertaModel
# import torch
#
# phobert_model = RobertaModel.from_pretrained('vinai/phobert-base', type_vocab_size=2)
# embedding = phobert_model.embeddings.word_embeddings
# token_type_embedding = phobert_model.embeddings.token_type_embeddings
# print(embedding.num_embeddings, embedding.embedding_dim)
# print(token_type_embedding.num_embeddings, token_type_embedding.embedding_dim)

# input_ids = torch.tensor([[0, 4473, 18, 646, 221, 4, 221, 6143, 6, 86, 7, 16, 18, 4, 221, 2044, 6, 1116, 18395, 4,
#                            38207, 2, 2446, 1829, 32054, 104, 366, 471, 104, 90, 129, 277, 32, 52, 1021, 23, 2, 1, 1, 1,
#                            1, 1, 1],
#                           [0, 51536, 646, 221, 328, 9, 2887, 4, 14110, 1124, 4, 12177, 12826, 678, 4, 275, 45554, 13143,
#                            4, 531, 2167, 1251, 8410, 4, 1501, 4, 2065, 2, 2446, 1829, 32054, 104, 366, 471, 104, 90,
#                            129, 277, 32, 52, 1021, 23, 2],
#                           [0, 4473, 116, 646, 9, 271, 926, 4, 36167, 1111, 4, 1046, 4, 116, 837, 4, 9141, 4, 116, 2229,
#                            2, 2446, 1829, 32054, 104, 366, 471, 104, 90, 129, 277, 32, 52, 1021, 23, 2, 1, 1, 1, 1, 1,
#                            1, 1],
#                           [0, 4473, 14110, 646, 5267, 4, 221, 2877, 4, 6146, 4, 1674, 4, 110, 201, 2, 2446, 1829, 32054,
#                            104, 366, 471, 104, 90, 129, 277, 32, 52, 1021, 23, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
#                          dtype=torch.long)
#
# print(torch.max(input_ids))
#
# input_type_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
#                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
#                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long)
#
# attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
#                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
#                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long)
#
# output = phobert_model(input_ids, token_type_ids=input_type_ids, attention_mask=attention_mask)
# print(output)


from models.phobert_query_ner import PhoBertQueryNER
from models.query_ner_config import PhobertQueryNerConfig

config = PhobertQueryNerConfig.from_pretrained('vinai/phobert-base',
                                               mrc_dropout=0.1, )
model = PhoBertQueryNER.from_pretrained('vinai/phobert-base', config=config)
