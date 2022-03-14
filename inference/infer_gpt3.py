import argparse
import json
from tqdm import tqdm
import numpy as np
import torch
import transformers
from utils.constants import NUMERSENSE_ANSWERS
import openai
from utils.constants import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

def checker(args, answer, pred):
    if args.task == 'numersense':
        if answer == pred:
            return 1
        if answer in ['no', 'zero'] and pred in ['no', 'zero']:
            return 1
        return 0
    return 1 if answer == pred else 0

def score_for_input(args, query, cands, knowledge=None):
    prompts = None
    prefix = '<|endoftext|>' if knowledge is None else knowledge
    if args.task == 'numersense':
        prompts = [f'{prefix}{query.replace("<mask>", cand)}' for cand in cands]

    if prompts is None:
        raise Exception(f'score_for_input() not implemented for {args.task}!')

    while True:
        try:
            response = openai.Completion.create(
                engine='davinci',
                prompt=prompts,
                max_tokens=0, # suppress continuation
                # temperature=1.,
                # top_p=0.5,
                # stop='\n',
                logprobs=0,
                echo=True, # so the logprobs of prompt tokens are shown
            )
            break
        except Exception as e:
            print(e)
            import time
            time.sleep(1)

    scores = []
    for c, cand in enumerate(cands):
        query_offset_tokens = 0
        while response['choices'][c]['logprobs']['text_offset'][query_offset_tokens] < len(prefix):
            query_offset_tokens += 1
        logprobs = response['choices'][c]['logprobs']['token_logprobs'][query_offset_tokens:]
        if args.average_loss:
            score = np.mean(logprobs)
        else:
            score = np.sum(logprobs)
        scores.append(score)
    scores = torch.tensor(scores)
    probs = torch.softmax(scores, dim=0)
    return scores, probs

def score_for_query(args, query, cands, knowledges=[]):
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
    scores, probs = score_for_input(args, query, cands)
    scores_.append(scores)
    probs_.append(probs)

    # with knowledge
    if len(knowledges) > 0:
        for i in range(0, v * h, h):
            knowledge = ' '.join(knowledges[i:i+h])
            scores, probs = score_for_input(args, query, cands, knowledge)
            scores_.append(scores)
            probs_.append(probs)

    return torch.stack(scores_), torch.stack(probs_)

def process_item(args, item):
    query = item['query']
    if 'cands' in item:
        cands = item['cands']
    elif args.task == 'numersense':
        cands = NUMERSENSE_ANSWERS
    else:
        raise Exception('process_item() not implemented for {args.task}!')

    knowledges = item['knowledges'] if 'knowledges' in item else []
    scores_, probs_ = score_for_query(args, query, cands, knowledges)
    scores, _ = torch.max(scores_, dim=0)
    probs, _ = torch.max(probs_, dim=0)

    if args.aggfunc == 'best_score':
        p = scores.argmax().item()
    elif args.aggfunc == 'best_prob':
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

    if args.task == 'numersense' and args.submit:
        item['output'] = {
            'probe': query,
            'result_list': [{'word': _[1]} for _ in sorted(zip(probs.tolist(), NUMERSENSE_ANSWERS), reverse=True)],
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['numersense'])
    parser.add_argument('--model-type', type=str, default='gpt3')
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--average-loss', action='store_true')
    parser.add_argument('--h', type=int, default=1)
    parser.add_argument('--v', type=int, default=-1)
    parser.add_argument('--aggfunc', type=str, default='best_prob', choices=['best_score', 'best_prob'])
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--n', type=int, default=None)
    args = parser.parse_args()
    args.output_path = f'data/{args.task}/inference/inference_{args.model_type.split("/")[-1]}.{args.input_path.split("/")[-1]}'

    if args.interactive:
        while True:
            example = input(f'Enter a {args.task} example: ')
            if args.task == 'csqa':
                splits = example.split('\t')
                query, cands = splits[0], splits[1:]
                item = {'query': query, 'cands': cands}
                process_item(args, item)
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
        process_item(args, item)
        if 'ok' in item:
            num += item['ok']
            den += 1
            pbar.set_postfix({'acc': num / den})

    if args.task == 'numersense' and args.submit:
        with open(args.output_path, 'w') as f:
            for item in ds:
                json.dump(item['output'], f)
                f.write('\n')
    else:
        with open(args.output_path, 'w') as f:
            json.dump(ds, f, indent=4)

if __name__ == '__main__':
    main()

