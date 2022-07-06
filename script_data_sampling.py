"""
EXECUTION: python script_data_sampling.py
           --percent 10

PURPOSE: for each combination of <category> & <language> (as specified in DATA_WEIGHTS.csv), the script
         - reads the original data file at <data_original>/<category>_<language>.jsonl
         - samples <percent>% of the data
         - writes the sampled data file at <data_sampled>/<category>_<language>_<percent>p.jsonl
"""
import argparse
import os
from src.sampling import get_file_path, read_data_weights
from itertools import product
from os.path import isfile, dirname


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
    # 1. read DATA_WEIGHTS.csv
    categories, languages, data_weights, data_weights_sampling = read_data_weights(percent=args.percent,
                                                                                   verbose=args.verbose)

    for category, language in product(categories, languages):
        weight = data_weights_sampling[category][language]
        print(f"> category = {category}, language = {language}, weight = {weight}")

        if weight > 0:
            # 2. make sure that all original data files exist
            file_path_original = get_file_path(category, language)
            assert isfile(file_path_original), \
                f"ERROR! file for category = {category}, language = {language} does not exist at {file_path_original}"

            # 3. make sure that <data_sampled> folder exists
            file_path_sampled = get_file_path(category, language, original=False, percent=args.percent)
            os.makedirs(dirname(file_path_sampled), exist_ok=True)

            # 4. sample <percent>% of the original data and write
            number_of_original_documents = sum(1 for _ in open(file_path_original))
            number_of_sampled_documents = int(weight*number_of_original_documents)

            with open(file_path_original) as infile, open(file_path_sampled, 'w') as outfile:
                for line in reservoir_sampling(infile, number_of_sampled_documents):
                    outfile.write(line)

            if args.verbose:
                print(f".. from {number_of_original_documents} original documents, "
                      f"wrote {number_of_sampled_documents} sampled documents to {file_path_sampled}")
                print()
        else:
            if args.verbose:
                print(f".. skipped")
                print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--percent", type=int, default=10)
    parser.add_argument("--verbose", action='store_true')
    _args = parser.parse_args()

    main(_args)
