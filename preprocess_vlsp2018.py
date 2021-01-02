import os
from utils.read_file import *
from vncorenlp import VnCoreNLP

annotator = VnCoreNLP(address="http://127.0.0.1", port=9000) 

labels = ['O', 'PER', 'LOC', 'ORG', 'MIS']
entity_labels = labels[1:]
map2true_labels = {
    'MIS': 'MISC',
}
label2idx = {map2true_labels.get(label, label): i for i, label in enumerate(labels)}

label2query = {
    'LOC': "thực thể địa danh bao gồm tên gọi các hành tinh, thực thể tự nhiên, địa lí lịch sử, vùng quần cư, công trình kiến trúc xây dụng, địa điểm, địa chỉ",
    'PER': "thực thể người bao gồm tên, tên đệm và họ của người, tên động vật, nhân vật hư cấu, bí danh",
    'ORG': "thực thể tổ chức bao gồm các cơ quan chính phủ, công ty, thương hiệu, tổ chức chính trị, ấn phẩm, tổ chức công cộng",
    'MISC': "thực thể bao gồm quốc tịch, ngôn ngữ, môn học, danh hiệu, cuộc thi",
}

for k in label2query.keys():
    label2query[k] = ' '.join(sum(annotator.tokenize(label2query[k]), []))


def process_sentence(context: str, positions: dict, sample_idx):
    res = []
    for label in entity_labels:
        poses = positions.get(label, [])
        new_poses = [(s, e - 1) for s, e in poses]
        label = map2true_labels.get(label, label)
        qas_id = f'{sample_idx}.{label2idx[label]}'
        start_position, end_position = zip(*new_poses) if len(new_poses) > 0 else ([], [])
        span_position = [f"{s};{e}" for s, e in new_poses]
        res.append({
            'qas_id': qas_id,
            'context': context,
            'query': label2query[label],
            'span_position': span_position,
            'start_position': start_position,
            'end_position': end_position,
            'impossible': len(span_position) == 0,
            'entity_label': label,
        })

    return res


def process_sentences(sentences):
    res = []
    for i, sentence in enumerate(sentences):
        res.extend(process_sentence(
            context=sentence['context'],
            positions=sentence['positions'],
            sample_idx=i,
        ))
    return res


def process(jsonl_fn, output_fn):
    sentences = load_jsonl(jsonl_fn)
    processed_sentences = process_sentences(sentences)
    list2jsonl(processed_sentences, output_fn)


if __name__ == '__main__':
    input_dir = 'data/vlsp2018/preprocessed_data'
    input_fns = ['train.jsonl', 'dev.jsonl', 'test.jsonl', ]
    output_dir = 'data/vlsp_processed'
    output_fns = ['train.jsonl', 'dev.jsonl', 'test.jsonl']

    for input_fn, output_fn in zip(input_fns, output_fns):
        process(os.path.join(input_dir, input_fn), os.path.join(output_dir, output_fn))
