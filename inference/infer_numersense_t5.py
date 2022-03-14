import argparse
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from utils.constants import NUMERSENSE_ANSWERS


def _score_cands(tokenizer, model, source, cands):
    with torch.no_grad():
        input_ids = tokenizer(source.replace('<mask>', '<extra_id_0>'), return_tensors='pt').input_ids.cuda()
        labels_ten = tokenizer('<extra_id_0> %s <extra_id_1>' % cands[-1], return_tensors='pt').input_ids.cuda()
        logits = model(input_ids=input_ids, labels=labels_ten).logits  # (B=1, L, V)
        labels = tokenizer(['<extra_id_0> %s <extra_id_1>' % cand for cand in cands[:-1]], return_tensors='pt').input_ids.cuda()  # (B, L)
        scores = logits[0, 1, labels[:, 1]].tolist()
        if labels_ten.size(1) == 5:
            score_ten = (logits[0, 1, labels_ten[0, 1]].item() + logits[0, 2, labels_ten[0, 2]].item()) / 2
        else:
            score_ten = logits[0, 1, labels_ten[0, 1]].item()
        scores.append(score_ten)
        return torch.tensor(scores)


def score_cands(tokenizer, model, source):
    scores1 = _score_cands(tokenizer, model, source, NUMERSENSE_ANSWERS)
    scores2 = _score_cands(tokenizer, model, source, [cand.capitalize() for cand in NUMERSENSE_ANSWERS])
    scores = (scores1 + scores2) / 2
    probs = F.softmax(scores, dim=0)
    return scores, probs


def scores_for_query(tokenizer, model, query, knowledges, h, v):
    if h == -1 and v == -1:
        raise Exception('h and v cannot be both -1!')
    n = len(knowledges)
    if n % h or n % v:
        raise Exception('h and v must be divisible by n!')
    if h == -1:
        h = n // v
    if v == -1:
        v = n // h
    scores_, probs_ = [], []
    scores, probs = score_cands(tokenizer, model, query)
    scores_.append(scores)
    probs_.append(probs)
    for i in range(0, v*h, h):
        source = ' '.join(knowledges[i:i+h] + [query])
        scores, probs = score_cands(tokenizer, model, source)
        scores_.append(scores)
        probs_.append(probs)
    return torch.stack(scores_), torch.stack(probs_)


def cmp(answer, pred):
    if answer == pred:
        return 1
    if answer in ['no', 'zero'] and pred in ['no', 'zero']:
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, default='t5-11b', choices=['t5-small', 't5-base', 't5-large', 't5-3b', 't5-11b'])
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--h', type=int, default=1)
    parser.add_argument('--v', type=int, default=-1)
    parser.add_argument('--aggfunc', type=str, default='bestprob', choices=['poe', 'moe', 'bestprob'])
    parser.add_argument('--interactive', action='store_true')
    args = parser.parse_args()
    args.output_path = f'data/numersense/inference/inference_{args.model_type.split("/")[-1]}.{args.input_path.split("/")[-1]}'

    tokenizer = transformers.T5Tokenizer.from_pretrained(args.model_type)
    model = transformers.T5ForConditionalGeneration.from_pretrained(args.model_type)
    model.cuda()
    model.eval()

    if args.interactive:
        while True:
            example = input(f'Enter a numersense example: ')
            scores, probs = score_cands(tokenizer, model, example)
            p = probs.argmax().item()
            pred = NUMERSENSE_ANSWERS[p]
            conf = probs[p].item()
            print(pred, conf)
            print(probs.tolist())

    with open(args.input_path) as f:
        ds = json.load(f)

    pbar = tqdm(ds)
    num, den = 0, 0
    for item in pbar:
        query = item['query']

        if 'knowledges' not in item:
            source = query
            scores, probs = score_cands(tokenizer, model, source)
        else:
            knowledges = item['knowledges']
            scores_, probs_ = scores_for_query(tokenizer, model, query, knowledges, args.h, args.v)
            if args.aggfunc == 'poe':
                probs = torch.prod(probs_, dim=0)
            elif args.aggfunc == 'moe':
                probs = torch.sum(probs_, dim=0)
            elif args.aggfunc == 'bestprob':
                probs, _ = torch.max(probs_, dim=0)
            item['scores_mat'] = scores_.tolist()
            item['probs_mat'] = probs_.tolist()
            item['agg_probs'] = probs.tolist()

        p = probs.argmax().item()
        pred = NUMERSENSE_ANSWERS[p]
        item['pred'] = pred

        if 'answer' in item:
            answer = item['answer']
            ok = cmp(answer, pred)
            item['ok'] = ok

        item['output'] = {
            'probe': query,
            'result_list': [{'word': _[1]} for _ in sorted(zip(probs.tolist(), NUMERSENSE_ANSWERS), reverse=True)],
        }

        if 'ok' in item:
            num += item['ok']
            den += 1
            pbar.set_postfix({'acc': num / den})

    if args.submit:
        with open(args.output_path, 'w') as f:
            for item in ds:
                json.dump(item['output'], f)
                f.write('\n')
    else:
        with open(args.output_path, 'w') as f:
            json.dump(ds, f, indent=4)
#        with open('out.txt', 'w') as f:
#            for item in ds:
#                f.write('%d\n' % item['ok'])

if __name__ == '__main__':
    main()

