from utils.gpt3_generation import request
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
import json
import torch
import click
from pathlib import Path
from typing import List


def prompt_format(prompt_path: str, keywords: List[str], query: str):
    with open(prompt_path) as f:
        context_string = f.read().strip('\n')
    if keywords is not None:
        n = np.random.choice(range(1, len(keywords)+1))      # number of keywords
        keywords = random.sample(keywords, n)                # subset of keywords
        context_string = context_string.replace('{keywords}', ', '.join(keywords))
    if query is not None:
        context_string = context_string.replace('{question}', query)
    return context_string


@click.command()
@click.option('--task', type=str, default=None)
@click.option('--input_path', type=str, default=None)
@click.option('--output_path', type=str, default=None)
@click.option('--prompt_path', type=str, default=None)
@click.option('--num_knowledge', type=int, default=20)
@click.option('--top_p', default=0.5, type=float)
@click.option('--temperature', default=1.0, type=float)
@click.option('--max_tokens', default=60, type=int)
@click.option('--n', default=None, type=int)
def main(
    task: str,
    input_path: str,
    output_path: str,
    prompt_path: bool,
    num_knowledge: int,
    top_p: float,
    temperature: float,
    max_tokens: int,
    n: int,
):
    # read examples for inference
    eval_df = pd.read_json(input_path)
    eval_df = eval_df[:n]

    # generate knowledge!
    generated_examples = []

    for i, row in tqdm(eval_df.iterrows(), total=n):
        context_string = prompt_format(
            prompt_path,
            keywords=None,
            query=row['query'])
        knowledges = request(
            context_string,
            n=num_knowledge,
            top_p=top_p,
            temperature=temperature,
            max_tokens=max_tokens)

        row['knowledges'] = list(set(knowledges))
        generated_examples.append(row.to_dict())

    with open(output_path, 'w') as fo:
        json.dump(generated_examples, fo, indent=4)


if __name__ == '__main__':
    main()
