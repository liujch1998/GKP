import json

split2path = {
    'train': '../data/csqa2/raw/CSQA2_train.json',
    'dev': '../data/csqa2/raw/CSQA2_dev.json',
    'test': '../data/csqa2/raw/CSQA2_test_no_answers.json',
}

def process(s):
    s = s.strip(' ')
    s = s[0].upper() + s[1:]
    if s[-1] not in ['.', '?']:
        s += '.'
    return s

for split, path in split2path.items():
    with open(path) as f:
        ds = []
        for line in f:
            js = json.loads(line)
            query = js['question']
            query = process(query)
            item = {'query': query}
            if split in ['train', 'dev']:   # test data does not have answers
                item['answer'] = js['answer']
            if js['relational_prompt_used']:
                item['relational_prompt'] = js['relational_prompt']
            else:
                item['relational_prompt'] = None
            if js['topic_prompt_used']:
                item['topic_prompt'] = js['topic_prompt']
            else:
                item['topic_prompt'] = None
            ds.append(item)
    with open('../data/csqa2_test/%s.csqa2.json' % split, 'w') as f:
        json.dump(ds, f, indent=4)

