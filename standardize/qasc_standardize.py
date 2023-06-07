import json

split2path = {
    'train': '../data/qasc/raw/train.jsonl',
    'dev': '../data/qasc/raw/dev.jsonl',
    'test': '../data/qasc/raw/test.jsonl',
}

def process(s, capitalize=False, add_period=False):
    if capitalize:
        s = s.capitalize()
    if add_period:
        if s[-1] != '.':
            s += '.'
    return s

for split, path in split2path.items():
    with open(path) as f:
        ds = []
        for line in f:
            js = json.loads(line)
            query = js['question']['stem']
            query = process(query, capitalize=True)
            cands = [_['text'] for _ in js['question']['choices']]
            item = {'query': query, 'cands': cands}
            if 'answerKey' in js:
                answer = cands[ord(js['answerKey']) - ord('A')]
                item['answer'] = answer
            facts = []
            for i in range(1, 3):
                if f'fact{i}' in js:
                    facts.append(process(js[f'fact{i}'], capitalize=True, add_period=True))
            item['facts'] = facts
            combinedfacts = []
            if 'combinedfact' in js:
                combinedfacts.append(process(js['combinedfact'], capitalize=True, add_period=True))
            item['combinedfacts'] = combinedfacts
            ds.append(item)
    with open('../data/qasc_test/%s.qasc.json' % split, 'w') as f:
        json.dump(ds, f, indent=4)

