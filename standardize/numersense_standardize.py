import json

split2path = {
    'train': '../ds/numersense/train.masked.tsv',
    'validation': '../ds/numersense/validation.masked.tsv',
    'test.all': '../ds/numersense/test.all.masked.txt',
    'test.core': '../ds/numersense/test.core.masked.txt',
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
    with open('../ds/numersense/%s.json' % split, 'w') as f:
        json.dump(ds, f, indent=4)

