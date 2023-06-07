import json

split2path = {
    'train': '../data/csqa/raw/train_rand_split.jsonl',
    'dev': '../data/csqa/raw/dev_rand_split.jsonl',
    'test': '../data/csqa/raw/test_rand_split_no_answers.jsonl',
}

for split, path in split2path.items():
    with open(path) as f:
        ds = []
        for line in f:
            js = json.loads(line)
            query = js['question']['stem']
            cands = [_['text'] for _ in js['question']['choices']]
            item = {'query': query, 'cands': cands}
            if 'answerKey' in js:
                answer = cands[ord(js['answerKey']) - ord('A')]
                item['answer'] = answer
            ds.append(item)
    with open('../data/csqa_test/%s.csqa.json' % split, 'w') as f:
        json.dump(ds, f, indent=4)

