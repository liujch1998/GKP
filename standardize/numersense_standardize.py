import json

split2path = {
    'train': 'data/numersense/raw/train.masked.tsv',
    'validation': 'data/numersense/raw/validation.masked.tsv',
    'test.all': 'data/numersense/raw/test.all.masked.txt',
    'test.core': 'data/numersense/raw/test.core.masked.txt',
}

for split, path in split2path.items():
    with open(path) as f:
        ds = []
        for line in f:
            if split in ['train', 'validation']:
                [query, answer] = line.strip('\n').split('\t')
                ds.append({'query': query, 'answer': answer})
            else:
                query = line.strip('\n').strip(' ')
                ds.append({'query': query})
    with open('data/numersense/%s.json' % split, 'w') as f:
        json.dump(ds, f, indent=4)

