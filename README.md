# Generated Knowledge Prompting

This repository contains the code for our ACL 2022 paper, [Generated Knowledge Prompting for Commonsense Reasoning](https://liujch1998.github.io/assets/papers/GKP_v8.pdf).

## Installation

```
conda env create -f environment.yml
conda activate gkp
```

## Download and Standardize the data

First, download the dataset for the following tasks: [NumerSense](https://github.com/INK-USC/NumerSense), [CommonsenseQA](https://www.tau-nlp.org/commonsenseqa), [CommonsenseQA 2.0](https://github.com/allenai/csqa2), and [QASC](https://allenai.org/data/qasc).

Next, use `standardize/{task_name}_standardize.py` to put the data in a unified format.

For convenience, we provide the standardized data at `data/{task_name}/{split_name}.{task_name}.json`

## Generate Knowledge

Use `gpt3_generate_knowledge.py` to generate knowledge for a task dataset.
For example, to generate knowledge for the validation set of NumerSense, run
```
python knowledge/gpt3_generate_knowledge.py \
    --task numersense \
    --input_path data/numersense/validation.json \
    --output_path data/numersense/knowledge/knowledge_gpt3.validation.json \
    --prompt_path knowledge/prompts/numersense_prompt.txt
```
This will create a JSON file that contains the knowledge statements for each question in this dataset.

This step needs access to GPT-3.
For convenience, we provide the knowledge files that were produced by our method and were used in downstream evaluation.
These files are at `data/{task_name}/knowledge/knowledge_gpt3.{split_name}.{task_name}.json`

## Inference with Knowledge Prompting

For the NumerSense task, use `infer_numersense_t5.py` to run inference with the generated knowledge.
For example:
```
CUDA_VISIBLE_DEVICES=0 python inference/infer_numersense_t5.py \
    --model-type t5-11b \
    --input-path data/numersense/knowledge/knowledge_gpt3.validation.json
```
where `--input-path` is the knowledge file we produced in the previous step at the `--output-path`.
This will create a JSON file that contains the inference results under `data/{task_name}/inference/`.

For the CommonsenseQA, CommonsenseQA 2.0, and QASC tasks, use `infer_t5.py` instead.
For example:
```
CUDA_VISIBLE_DEVICES=0 python inference/infer_t5.py \
    --task csqa \
    --model-type allenai/unifiedqa-t5-11b \
    --input-path data/csqa/knowledge/knowledge_gpt3.dev.csqa.json
```

To get the baseline of your inference model (i.e. without using generated knowledge), use the standardized data file as `--input-path`:
```
CUDA_VISIBLE_DEVICES=0 python inference/infer_t5.py \
    --task csqa \
    --model-type allenai/unifiedqa-t5-11b \
    --input-path data/csqa/dev.csqa.json
```

Please check the argparse choices in the scripts for more parameters available.

## A special note on the QASC dataset

On the QASC dataset, we find that the text-infilling question format is more suitable when using the off-the-shelf T5 as inference model.
Therefore, we first convert the dataset to an infilling format:
```
python standardize/qasc_to_infill.py
```
For convenience, we include the infilling-format QASC dataset at `data/qasc/{split_name}.qasc_infill.json`.

For the knowledge generation step, still use the original standardized data file.
For the inference step, use the knowledge file where questions are in the text-infilling format:
```
CUDA_VISIBLE_DEVICES=0 python inference/infer_t5.py \
    --task qasc \
    --model-type t5-11b \
    --input-path data/qasc/knowledge/knowledge_gpt3.dev.qasc_infill.json
```

Note that it is not needed to use the infilling-format data with finetuned T5 or any UnifiedQA inference models.

## Citation

If you find this repository helpful, please consider citing our paper:
```
@article{liu2021generated,
  title={Generated knowledge prompting for commonsense reasoning},
  author={Liu, Jiacheng and Liu, Alisa and Lu, Ximing and Welleck, Sean and West, Peter and Bras, Ronan Le and Choi, Yejin and Hajishirzi, Hannaneh},
  journal={arXiv preprint arXiv:2110.08387},
  year={2021}
}
```

