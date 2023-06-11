"""
EXECUTION: python script_data_sampling.py
           --percent 10

PURPOSE: for each combination of <category> & <language> (as specified in SAMPLING_WEIGHTS.csv), the script
         - reads the original data file at <data_original>/<category>_<language>.jsonl
         - samples <percent>% of the data
         - writes the sampled data file at <data_train>/<category>_<language>_<percent>p.jsonl
"""
import argparse
import os
from itertools import product
from os.path import isfile, dirname, getsize
import time

from src.env import Env
from src.sampling import get_file_path, read_sampling_weights
from scripts.data_helpers.script_concatenate_data_by_language import concatenate_data_by_language


import random


def reservoir_sampling(l, k):
    """
    taken from
    https://stackoverflow.com/questions/40144869/python-read-random-lines-from-a-very-big-file-and-append-to-another-file
    """
    it = iter(l)
    try:
        result = [next(it) for _ in range(k)]  # use xrange if on python 2.x
    except StopIteration:
        raise ValueError("Sample larger than population")

    for i, item in enumerate(it, start=k):
        s = random.randint(0, i)
        if s < k:
            result[s] = item

    # random.shuffle(result)  # additional cost without effect
    return result


def main(args):
    env = Env()

    # 1. read SAMPLING_WEIGHTS.csv
    categories, languages, sampling_weights, sampling_weights_sampling = read_sampling_weights(percent=args.percent,
                                                                                               verbose=args.verbose)

    for category, language in product(categories, languages):
        weight = sampling_weights_sampling[category][language]

        if weight > 0:
            ts = time.time()
            print(f"> category = {category}, language = {language}, weight = {weight}")

            # 2. make sure that all original data files exist
            file_path_original = get_file_path(category,
                                               language,
                                               kind="data_original")
            assert isfile(file_path_original), \
                f"ERROR! file for category = {category}, language = {language} does not exist at {file_path_original}"
            file_size_original = getsize(file_path_original)
            print(f".. size = {file_size_original/float(10**6):.1f} MB -> ", end="")

            # 3. make sure that <data_train> folder exists
            file_path_sampled = get_file_path(category,
                                              language,
                                              kind="data_eval" if args.evaluation else "data_train",
                                              percent=args.percent)
            os.makedirs(dirname(file_path_sampled), exist_ok=True)

            # 4. sample <percent>% of the original data and write
            number_of_original_documents = sum(1 for _ in open(file_path_original))
            number_of_sampled_documents = int(weight*number_of_original_documents)

            with open(file_path_original) as infile, open(file_path_sampled, 'w') as outfile:
                for line in reservoir_sampling(infile, number_of_sampled_documents):
                    outfile.write(line)

            file_size_sampled = getsize(file_path_sampled)
            print(f"{file_size_sampled/float(10**6):.1f} MB ", end="")

            if args.verbose:
                print(f".. from {number_of_original_documents} original documents, "
                      f"wrote {number_of_sampled_documents} sampled documents to {file_path_sampled}")

            te = time.time()
            print(f"[time = {te-ts:.1f}s]")
            print()

        else:
            if args.verbose:
                print(f"> category = {category}, language = {language}, weight = {weight} .. skipped")
                print()

    # concatenate data by language (only if args.evaluation == 1)
    if args.evaluation:
        print("\n=================")
        print(f"> concatenate data by language in {env.data_eval}")
        concatenate_data_by_language(env.data_eval, inplace=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--percent", type=int, default=10)
    parser.add_argument("--evaluation", type=bool, default=0)
    parser.add_argument("--verbose", action='store_true')
    _args = parser.parse_args()

    main(_args)
