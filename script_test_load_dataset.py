"""
EXECUTION: python script_test_load_dataset.py
           [--dataset_files wiki_is.jsonl wiki_da.jsonl
            --batch_size 100000]

PURPOSE: the script
         - loads the data in <dataset_files> in batches of <batch_size>
         - uses the get_training_corpus generator to read it
         - prints information to check that everything works as expected
"""
import argparse
from datasets import load_dataset, Dataset
from typing import List
from src.helpers import get_training_corpus_combined


def get_training_corpus(_datasets: List[Dataset], batch_size: int):
    for dataset in _datasets:
        for i in range(0, len(dataset['train']), batch_size):
            yield str(dataset['train'][i: i + batch_size]["text"])


def main(args):
    dataset_files = args.dataset_files
    batch_size = args.batch_size

    # 0. Load Datasets
    datasets = [
        load_dataset('json',
                     data_files={'train': data_file},
                     # field=["text", "id"],
                     # features=features,
                     )
        for data_file in dataset_files
    ]

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
        else:
            print(f"> length dataset: {len(dataset['train'])}")
            generator = get_training_corpus_combined(dataset, batch_size)

        for elem in generator:
            print(type(elem), len(elem), elem[:10])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_files", nargs='+', type=str, default=["wiki_is.jsonl", "wiki_da.jsonl"])
    parser.add_argument("--batch_size", type=int, default=100000)
    _args = parser.parse_args()

    main(_args)
