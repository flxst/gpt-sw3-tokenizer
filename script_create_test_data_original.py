"""
EXECUTION: python script_test_data_original.py
           --number_of_documents 20

PURPOSE: for each combination of <category> & <language> (as specified in DATA_WEIGHTS.csv), the script
         - creates <number_of_documents> fake original documents for testing
         - writes the fake original data to <data_original>/<category>_<language>.jsonl
"""

import argparse
import os
from src.env import Env
import json
from src.sampling import get_file_path, read_data_weights


def main(args):
    env = Env()
    os.makedirs(env.data_original, exist_ok=True)

    # 1. read DATA_WEIGHTS.csv
    categories, languages, data_weights, _ = read_data_weights(verbose=True)

    # 2. write fake original data
    number_of_files = 0
    for category in categories:
        for language in languages:
            file_path_original = get_file_path(category, language)
            data = [f"Example {category}_{language} nr. {n}" for n in range(args.number_of_documents)]
            with open(file_path_original, "w") as file:
                for elem in data:
                    file.write(json.dumps({"text": elem}) + "\n")
            number_of_files += 1

    print(f"\n> wrote {number_of_files} files "
          f"({len(categories)} categories and {len(languages)} languages) "
          f"with {args.number_of_documents} fake documents each "
          f"[see {env.data_original}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--number_of_documents", type=int, default=20)
    _args = parser.parse_args()

    main(_args)
