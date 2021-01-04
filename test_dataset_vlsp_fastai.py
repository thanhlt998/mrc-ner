from datasets.mrc_ner_dataset_vlsp_fastai import SentenceTransform, YTransform, BeforeBatchTransform
from utils.read_file import load_jsonl
from transformers import PhobertTokenizer
from fastai.text.all import (
    Pipeline,
    Datasets,
    tensor,
)


def test_sentence_transform():
    data = load_jsonl('data/vlsp_processed/test.jsonl')
    sent_tfms = SentenceTransform(
        tokenizer=tokenizer,
        possible_only=False,
    )
    pipeline = Pipeline([sent_tfms])

    for o in data:
        if o['start_position']: break
    # o = data[0]
    new_o = pipeline(o)
    print(new_o, tokenizer.convert_ids_to_tokens(new_o['token_ids']))

    y_tfms = YTransform()
    pipeline1 = Pipeline([y_tfms])
    print(pipeline1(new_o))


def test_dataset():
    print("#" * 15, 'test dataset', "#" * 15)
    data = load_jsonl('data/vlsp_processed/test.jsonl')

    sent_tfms = SentenceTransform(
        tokenizer=tokenizer,
        possible_only=False,
    )
    data = [sent_tfms(i) for i in data[:1000]]
    x_tfms = [lambda x: x['token_ids'], tensor]
    y_tfms = [YTransform()]

    ds = Datasets(
        data,
        tfms=[x_tfms, y_tfms],
        n_inp=1,
    )

    print(ds[0])
    print(ds.show(ds[0]))
    print("#" * 15, 'end test dataset', "#" * 15)
    return ds


def test_dataloader():
    print("#" * 15, 'test dataloader', "#" * 15)
    ds = test_dataset()
    dl = ds.dataloaders(
        bs=4,
        before_batch=BeforeBatchTransform(
            max_seq_length=256,
            pad_fields=[0, 1, 2, 3, 4],
            pad_values=[tokenizer.pad_token_id, 0, 0, 0, 0],
            pad_token_id=tokenizer.pad_token_id,
            sep_token_id=tokenizer.sep_token_id,
        ),
        n_inp=1,
        verbose=5,
    )
    batch = dl.one_batch()
    print(len(batch))
    for t in batch: print(t.size())
    print("#" * 15, 'end test dataloader', "#" * 15)


if __name__ == '__main__':
    # test_sentence_transform()
    tokenizer = PhobertTokenizer.from_pretrained('vinai/phobert-base')
    # test_dataset()
    test_dataloader()
