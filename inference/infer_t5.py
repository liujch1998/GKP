import argparse
import json
from tqdm import tqdm
import numpy as np
import torch
import transformers

def score_for_input(args, tokenizer, model, query, cands, knowledge=None):
    source, targets = None, None
    if args.task == 'csqa':
        if 'unifiedqa-t5' in args.model_type or args.model_ckpt is not None:  # T5-ft, UnifiedQA, UnifiedQA-ft
            source = f'{query} \\n ' + ' '.join([f'({chr(ord("a") + i)}) {cand}' for i, cand in enumerate(cands)])
            if knowledge is not None:
                source = f'{source} \\n {knowledge}'
            targets = cands
        elif 't5' in args.model_type:  # T5
            source = query
            if knowledge is not None:
                source = f'{knowledge} {source}'
            targets = [f'<extra_id_0> {cand} <extra_id_1>' for cand in cands]
    elif args.task == 'csqa2':
        if 't5' in args.model_type and args.model_ckpt is not None:  # Unicorn
            source = query
            if knowledge is not None:
                source = f'{knowledge} {source}'
            targets = cands
        elif 't5' in args.model_type:  # T5
            source = f'It is <extra_id_0> that: {query}'
            if knowledge is not None:
                source = f'{knowledge} {source}'
            targets = [f'<extra_id_0> {cand} <extra_id_1>' for cand in ['true', 'false']]
    elif args.task == 'qasc':
        if 'unifiedqa-t5' in args.model_type or args.model_ckpt is not None:  # T5-ft, UnifiedQA, UnifiedQA-ft
            source = f'{query} \\n ' + ' '.join([f'({chr(ord("a") + i)}) {cand}' for i, cand in enumerate(cands)])
            if knowledge is not None:
                source = f'{source} \\n {knowledge}'
            targets = cands
        elif 't5' in args.model_type:  # T5
            source = query if '<extra_id_0>' in query else f'{query} <extra_id_0>' # to accomodate infilling input format
            if knowledge is not None:
                source = f'{knowledge} {source}'
            targets = [f'<extra_id_0> {cand} <extra_id_1>' for cand in cands]
    if source is None or targets is None:
        raise Exception(f'score_for_input() not implemented for {args.task} {args.model_type}!')

    scores = []
    input_ids = tokenizer(source, return_tensors='pt').input_ids.cuda()
    for i, cand in enumerate(cands):
        labels = tokenizer(targets[i], return_tensors='pt').input_ids.cuda()
        with torch.no_grad():
            loss = model(input_ids=input_ids, labels=labels).loss.item() # mean reduction
        if not args.average_loss:
            loss *= labels.size(1)
        score = -loss
        scores.append(score)
    scores = torch.tensor(scores)
    probs = torch.softmax(scores, dim=0)
    return scores, probs

def score_for_query(args, tokenizer, model, query, knowledges, cands):
    n = len(knowledges)
    h, v = args.h, args.v
    if h == -1 and v == -1:
        raise Exception('h and v cannot be both -1!')
    if h * v > n:
        raise Exception('h*v must be no larger than the number of knowledges!')
    if h == -1:
        h = n // v
    if v == -1:
        v = n // h

    scores_, probs_ = [], []

    # a pass w/o knowledge
    scores, probs = score_for_input(args, tokenizer, model, query, cands)
    scores_.append(scores)
    probs_.append(probs)

    # with knowledge
    if len(knowledges) > 0:
        for i in range(0, v * h, h):
            knowledge = ' '.join(knowledges[i:i+h])
            scores, probs = score_for_input(args, tokenizer, model, query, cands, knowledge)
            scores_.append(scores)
            probs_.append(probs)

    return torch.stack(scores_), torch.stack(probs_)

def checker(args, answer, pred):
    return 1 if answer == pred else 0

def process_item(args, tokenizer, model, item):
    query = item['query'] if 'query' in item else item['question']
    if 'cands' in item:
        cands = item['cands']
    elif args.task == 'csqa2':
        cands = ['yes', 'no']
    else:
        raise Exception('process_item() not implemented for {args.task}!')

    knowledges = item['knowledges'] if 'knowledges' in item else []
    scores_, probs_ = score_for_query(args, tokenizer, model, query, knowledges, cands)
    scores, _ = torch.max(scores_, dim=0)
    probs, _ = torch.max(probs_, dim=0)

    if args.aggfunc == 'best_score':
        p = scores.argmax().item()
    elif args.aggfunc == 'best_prob':
        p = probs.argmax().item()
    elif args.aggfunc == 'poe':
        probs = torch.prod(probs_, dim=0)
        p = probs.argmax().item()
    elif args.aggfunc == 'moe':
        probs = torch.sum(probs_, dim=0)
        p = probs.argmax().item()
    pred = cands[p]

    item['scores_'] = scores_.tolist()
    item['probs_'] = probs_.tolist()
    item['scores'] = scores.tolist()
    item['probs'] = probs.tolist()
    item['pred'] = pred

    if 'answer' in item:
        answer = item['answer']
        ok = checker(args, answer, pred)
        item['ok'] = ok

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['csqa', 'csqa2', 'qasc'])
    parser.add_argument('--model-type', type=str, required=True, choices=['t5-small', 't5-base', 't5-large', 't5-3b', 't5-11b', 'allenai/unifiedqa-t5-large', 'allenai/unifiedqa-t5-3b', 'allenai/unifiedqa-t5-11b'])
    parser.add_argument('--model-ckpt', type=str, default=None)
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--average-loss', action='store_true')
    parser.add_argument('--h', type=int, default=1)
    parser.add_argument('--v', type=int, default=-1)
    parser.add_argument('--aggfunc', type=str, default='best_prob', choices=['best_score', 'best_prob', 'poe', 'moe'])
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--n', type=int, default=None)
    args = parser.parse_args()
    args.output_path = f'data/{args.task}/inference/inference_{"" if args.model_ckpt is None else "ft"}{args.model_type.split("/")[-1]}.{args.input_path.split("/")[-1]}'

    tokenizer = transformers.T5Tokenizer.from_pretrained(args.model_type)
    model = transformers.T5ForConditionalGeneration.from_pretrained(args.model_ckpt if args.model_ckpt is not None else args.model_type)
    model.cuda()
    model.eval()

    if args.interactive:
        while True:
            example = input(f'Enter a {args.task} example: ')
            if args.task == 'csqa':
                splits = example.split(' -- ')
                query, cands = splits[0], splits[1:]
                item = {'query': query, 'cands': cands}
                process_item(args, tokenizer, model, item)
                print(item['pred'], item['probs'])
            elif args.task == 'csqa2':
                item = {'query': example}
                process_item(args, tokenizer, model, item)
                print(item['pred'], item['probs'])
            else:
                raise Exception(f'Interactive mode not implemented for {args.task}')

    with open(args.input_path) as f:
        ds = json.load(f)
        if args.n is not None:
            ds = ds[:args.n]

    pbar = tqdm(ds)
    num, den = 0, 0
    for item in pbar:
        process_item(args, tokenizer, model, item)
        if 'ok' in item:
            num += item['ok']
            den += 1
            pbar.set_postfix({'acc': num / den})

    with open(args.output_path, 'w') as f:
        json.dump(ds, f, indent=4)

if __name__ == '__main__':
    main()

