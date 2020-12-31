import json


def load_jsonl(fn):
    res = []
    with open(fn, mode='r', encoding='utf8') as f:
        for line in f:
            res.append(json.loads(line))
    return res


def list2jsonl(l: list, fn: str):
    with open(fn, mode='w', encoding='utf8') as f:
        for i in l:
            f.write(json.dumps(i, ensure_ascii=False))
            f.write('\n')
