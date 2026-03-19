import json
import re
import os

from verl.utils.hdfs_io import copy, makedirs
import argparse



from datasets import concatenate_datasets, load_dataset

#from rllm.data.dataset import DatasetRegistry
#from rllm.data.utils import fetch_live_code_bench_system_prompt
from system_prompts import LCB_SYSTEM_MESSAGE_GENERIC, LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE, LCB_FORMATTING_WITHOUT_STARTER_CODE

def fetch_live_code_bench_system_prompt(prompt: str, starter_code: str | None = None):
    # https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py
    prompt = LCB_SYSTEM_MESSAGE_GENERIC + "\n\n" + prompt
    if starter_code:
        prompt += f"### Format: {LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {LCB_FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def make_map_fn(split, data_source="agentica-org/DeepCoder-Preview-Dataset"):

    def process_fn(example, idx):
        starter_code = example.get("starter_code", "") or ""
        question = fetch_live_code_bench_system_prompt(
            example["problem"],
            starter_code if starter_code else None
        )

        tests_raw = example["tests"]
        tests = json.loads(tests_raw) if isinstance(tests_raw, str) else tests_raw

        # Convert TACO format (dict with inputs/outputs) before the list check
        if isinstance(tests, dict) and "inputs" in tests and "outputs" in tests:
            tests = [
                {"input": inp, "output": out, "testtype": "stdin_stdout"}
                for inp, out in zip(tests["inputs"], tests["outputs"], strict=False)
            ]

        if not isinstance(tests, list):
            tests = [tests] if tests else []

        # metadata can be None even when the key exists, so use `or {}`
        metadata = example.get("metadata") or {}
        if isinstance(metadata, str):
            metadata = json.loads(metadata) or {}

        fn_name = metadata.get("func_name", None)
        in_outs = {
            "inputs":  [t["input"]  for t in tests],
            "outputs": [t["output"] for t in tests],
        }
        if fn_name:
            in_outs["fn_name"] = str(fn_name)

        data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "code",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps(in_outs)
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "starter_code": starter_code,
            }
        }
        return data

    return process_fn


def prepare_deepcoder_data(train_size: int = None, test_size: int = None):
    train_dataset = concatenate_datasets([load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="primeintellect", split="train"), load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="taco", split="train"), load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="lcbv5", split="train")])
    test_dataset = concatenate_datasets([load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="codeforces", split="test"), load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="lcbv5", split="test")])

    if train_size:
        train_dataset = train_dataset.select(range(min(train_size, len(train_dataset))))
    if test_size:
        test_dataset = test_dataset.select(range(min(test_size, len(test_dataset))))

    # make_map_fn produces verl-compatible format:
    #   prompt: [{"role": "user", "content": ...}]
    #   reward_model: {"style": "rule", "ground_truth": '{"inputs": [...], "outputs": [...]}'}
    #   data_source: "agentica-org/DeepCoder-Preview-Dataset"
    train_dataset = train_dataset.map(make_map_fn("train"), with_indices=True, writer_batch_size=10, num_proc=16)
    test_dataset = test_dataset.map(make_map_fn("test"), with_indices=True, writer_batch_size=10, num_proc=16)

    return train_dataset, test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/opt/tiger/deepcoder')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    #num_few_shot = 5
    #data_source = 'openai/gsm8k'
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    train_dataset, test_dataset = prepare_deepcoder_data()
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    print(f"✅ 数据集已成功保存至: {local_dir}")
