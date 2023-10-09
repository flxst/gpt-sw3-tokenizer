"""
EXECUTION: python script_data_sampling.py
           --percent <percent>  # e.g. 10
           [--evaluation 0]     # 0 = <data_train>, 1 = <data_eval>

PURPOSE: for each combination of <category> & <language> (as specified in SAMPLING_WEIGHTS.csv), the script
         - reads the original data file at <data_original>/<category>_<language>.jsonl
         - samples <percent>% of the data
         - writes the sampled data file at <data_train>/<category>_<language>_<percent>p.jsonl
"""
import argparse
import os
import json
from itertools import product
from os.path import isfile, dirname, getsize, join
import time

from src.env import Env
from src.sampling import reservoir_sampling
from src.logger import Logger
from scripts.data_processing.script_concatenate_data_by_language import concatenate_data_by_language
from src.helpers import read_json


def main(args):
    env = Env()
    logger_folder = env.data_eval if args.evaluation else env.data_train
    logger = Logger(logger_folder)

    file_path_sample_indices = join(logger_folder, "SAMPLING.json")
    sample_indices = {}

    # 1. read SAMPLING_WEIGHTS.csv
    categories, languages, sampling_weights, sampling_weights_sampling = env.read_sampling_weights(percent=args.percent,
                                                                                                   verbose=env.verbose)
    logger.initialize(percent=args.percent, sampling_weights=sampling_weights)

    # 2. if evaluation, read SAMPLING.json from training for disjunct sampling
    if args.evaluation:
        train_sampling_path = join(env.data_train, "SAMPLING.json")
        train_indices = read_json(train_sampling_path)
    else:
        train_indices = {}

    for category, language in product(categories, languages):
        category_language = f"{category}_{language}.jsonl"
        exclude = train_indices[category_language] if args.evaluation else ()
        weight = sampling_weights_sampling[category][language]

        if weight > 0:
            ts = time.time()
            logger.log_print(f"> category = {category}, language = {language}, weight = {weight}")

            # 3. make sure that all source files in <data_original> exist
            file_path_original = env.get_file_path(category,
                                                   language,
                                                   kind="data_original")
            assert isfile(file_path_original), \
                f"ERROR! file for category = {category}, language = {language} does not exist at {file_path_original}"
            file_size_original = getsize(file_path_original)
            logger.log_print(f".. size = {file_size_original/float(10**6):.1f} MB -> ", end="")

            # 4. make sure that target folder <data_train> or <data_eval> exists
            file_path_sampled = env.get_file_path(category,
                                                  language,
                                                  kind="data_eval" if args.evaluation else "data_train")
            os.makedirs(dirname(file_path_sampled), exist_ok=True)

            # 5. sample <percent>% of the original data and write
            number_of_original_documents = sum(1 for _ in open(file_path_original))
            number_of_sampled_documents = int(weight*number_of_original_documents)

            with open(file_path_original) as infile:
                sample, sample_indices[category_language] = \
                    reservoir_sampling(infile, number_of_sampled_documents, exclude)
            with open(file_path_sampled, 'w') as outfile_sample:
                for line in sample:
                    outfile_sample.write(line)

            file_size_sampled = getsize(file_path_sampled)
            logger.log_print(f"{file_size_sampled/float(10**6):.1f} MB (ratio = {file_size_sampled/file_size_original:.2f})", end="")

            if env.verbose:
                logger.log_print(f".. from {number_of_original_documents} original documents, "
                                 f"wrote {number_of_sampled_documents} sampled documents to {file_path_sampled}")

            te = time.time()
            logger.log_print(f" [time = {te-ts:.1f}s]")
            logger.log_print()

        else:
            if env.verbose:
                logger.log_print(f"> category = {category}, language = {language}, weight = {weight} .. skipped")
                logger.log_print()

    # write sample indices
    with open(file_path_sample_indices, 'w') as outfile_sample_indices:
        json.dump(sample_indices, outfile_sample_indices)

    # concatenate data by language (only if args.evaluation == 1)
    if args.evaluation:
        logger.log_print("\n=================")
        logger.log_print(f"> concatenate data by language in {env.data_eval}")
        concatenate_data_by_language(env.data_eval, inplace=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--percent", type=int, default=10)
    parser.add_argument("--evaluation", type=bool, default=0)
    _args = parser.parse_args()

    main(_args)
