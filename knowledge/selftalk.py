from utils.gpt3_generation import request
from tqdm import tqdm
import json
import click
from typing import List

prefixes = [
    ('What is the definition of', 'The definition of _ is'),
    ('What is the main purpose of', 'The purpose of _ is to'),
    ('What is the main function of a', 'The main function of a _ is'),
    ('What are the properties of a', 'The properties of a _ are that'),
    ('What is a', '_ is'),
    ('What happened as a result of', 'As a result of _,'),
    ('What might have caused', 'The cause of _ was'),
]

@click.command()
@click.option('--input_path', type=str, default=None)
@click.option('--output_path', type=str, default=None)
@click.option('--n', default=None, type=int)
def main(
    input_path: str,
    output_path: str,
    n: int,
):
    with open(input_path) as f:
        ds = json.load(f)
        ds = ds[:n]

    for item in tqdm(ds):
        knowledges = []
        for (qp, ap) in prefixes:
            qcs = request(
                f'{item["query"]} {qp}' ,
                n=3,
                top_p=0.2,
                max_tokens=6,
                stop='?',
                engine='davinci',
            )
            for qc in qcs:
                acs = request(
                    f'{item["query"]} {qp} {qc} {ap.replace("_", qc)}',
                    n=1,
                    top_p=0.5,
                    max_tokens=10,
                    stop='.',
                    engine='davinci',
                )
                for ac in acs:
                    if ac[-1] != '.':
                        ac += '.'
                    knowledges.append(f'{ap.replace("_", qc)}, {ac}')
        item['knowledges'] = list(set(knowledges))

    with open(output_path, 'w') as f:
        json.dump(ds, f, indent=4)

if __name__ == '__main__':
    main()

