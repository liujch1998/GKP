from utils.gpt3_generation import request
from tqdm import tqdm
import numpy as np
import random
import json
import torch
import click
from typing import List


@click.command()
@click.option('--input_path', type=str, default=None)
@click.option('--output_path', type=str, default=None)
@click.option('--num_knowledge', type=int, default=20)
@click.option('--top_p', default=0.5, type=float)
@click.option('--temperature', default=1.0, type=float)
@click.option('--max_tokens', default=60, type=int)
@click.option('--n', default=None, type=int)
def main(
    input_path: str,
    output_path: str,
    num_knowledge: int,
    top_p: float,
    temperature: float,
    max_tokens: int,
    n: int,
):
    with open(input_path) as f:
        ds = json.load(f)
        ds = ds[:n]

    for item in tqdm(ds):
        knowledges = request(
            item['query'],
            #'<|endoftext|>',
            n=num_knowledge,
            top_p=top_p,
            temperature=temperature,
            max_tokens=max_tokens,
            stop='.')
        knowledges = [s.strip('\n') for s in knowledges]
        knowledges = [s.split('\n')[0] for s in knowledges]
        knowledges = [s.encode('ascii', 'ignore').decode().strip(' ') for s in knowledges]
        knowledges = [s for s in knowledges if s != '']
        knowledges = [s for s in knowledges if '<mask>' not in s]
        knowledges = [s if s[-1] in ['.', '?'] else s + '.' for s in knowledges]
        item['knowledges'] = list(set(knowledges))

    with open(output_path, 'w') as f:
        json.dump(ds, f, indent=4)

if __name__ == '__main__':
    main()

