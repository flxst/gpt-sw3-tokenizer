"""
EXECUTION: python script_test_load_dataset.py
            [--dataset_files wiki_is.jsonl wiki_da.jsonl]
            [--batch_size 100000]

PURPOSE: the script
         - loads the data in <data_original>/<dataset_files> in batches of <batch_size>
         - prints information to check that everything works as expected
"""
import argparse
from datasets import load_dataset, Dataset
from typing import List
from os.path import join

from os.path import abspath, dirname
import sys
BASE_DIR = abspath(dirname(dirname(dirname(abspath(__file__)))))
print(f">>> BASE_DIR: {BASE_DIR}")
sys.path.append(BASE_DIR)

from src.helpers import get_training_corpus_combined
from src.env import Env


def get_training_corpus(_datasets: List[Dataset], batch_size: int):
    for dataset in _datasets:
        for i in range(0, len(dataset['train']), batch_size):
            yield str(dataset['train'][i: i + batch_size]["text"])


def print_dataset_info(_identifier: str, _generator):
    for elem in _generator:
        print(f"{_identifier}: type = {type(elem)}, len = {len(elem)}, first 10 chars = '{elem[:10]}'")


def main(args):
    env = Env()
    dataset_files = [join(env.data_original, dataset_file) for dataset_file in args.dataset_files]
    batch_size = args.batch_size

    # 0. Load Datasets
    try:
        datasets = [
            load_dataset('json',
                         data_files={'train': data_file},
                         # field=["text", "id"],
                         # features=features,
                         )
            for data_file in dataset_files
        ]
    except FileNotFoundError:
        raise Exception(f"ERROR! dataset files = {dataset_files} not found")

    datasets_combined = load_dataset('json',
                                     data_files={'train': dataset_files},
                                     )

    for i, dataset in enumerate([datasets, datasets_combined]):
        print(f"\n=== {i} ===")
        print(f"> number of datasets: {len(dataset)}")

        if i == 0:
            for j in range(len(dataset)):
                print(f"> length dataset {j}: {len(dataset[j]['train'])}")
            generator = get_training_corpus(dataset, batch_size)
            print_dataset_info(f"single:", generator)
        else:
            print(f"> length dataset combined: {len(dataset['train'])}")
            generator = get_training_corpus_combined(dataset, batch_size)
            print_dataset_info("combined", generator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_files", nargs='+', type=str, default=["wiki_is.jsonl", "wiki_da.jsonl"])
    parser.add_argument("--batch_size", type=int, default=100000)
    _args = parser.parse_args()

    main(_args)
