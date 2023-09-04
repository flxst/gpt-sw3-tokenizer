"""
EXECUTION: python script_test_data_original.py
           [--number_of_documents 20]

PURPOSE: for each combination of <category> & <language> (as specified in SAMPLING_WEIGHTS.csv), the script
         - creates <number_of_documents> fake original documents for testing
         - writes the fake original data to <data_original>/<category>_<language>.jsonl
"""

import argparse
import os
import json

from os.path import abspath, dirname
import sys
BASE_DIR = abspath(dirname(dirname(dirname(abspath(__file__)))))
print(f">>> BASE_DIR: {BASE_DIR}")
sys.path.append(BASE_DIR)

from src.env import Env
from src.sampling import get_file_path, read_sampling_weights


def main(args):
    env = Env()
    os.makedirs(env.data_original, exist_ok=True)

    # 1. read SAMPLING_WEIGHTS.csv
    categories, languages, sampling_weights, _ = read_sampling_weights(verbose=True)

    # 2. write fake original data
    number_of_files = 0
    number_of_zero_weights = 0
    for category in categories:
        for language in languages:
            if sampling_weights[category][language] > 0:
                file_path_original = get_file_path(category,
                                                   language,
                                                   kind="data_original")
                data = [f"Example {category}_{language} nr. {n}" for n in range(args.number_of_documents)]
                with open(file_path_original, "w") as file:
                    for elem in data:
                        file.write(json.dumps({"text": elem}) + "\n")
                number_of_files += 1
            else:
                number_of_zero_weights += 1

    print(f"\n> wrote {number_of_files} files "
          f"({len(categories)} categories x {len(languages)} languages - {number_of_zero_weights} zero weights) "
          f"with {args.number_of_documents} fake documents each "
          f"[see {env.data_original}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--number_of_documents", type=int, default=20)
    _args = parser.parse_args()

    main(_args)
