"""Module that contains functions for sampling"""
import csv
import sys
import random
from typing import Tuple, List, Dict
from os.path import join
from src.env import Env

env = Env()


def get_file_path(category: str, language: str, kind: str) -> str:
    """
    get file path for data of kind 'kind', category 'category' and language 'language'

    Args:
        category: e.g. 'books'
        language: e.g. 'en'
        kind: e.g. 'data_original'

    Returns:
        file_path: e.g. '<data_original>/books_en.jsonl
    """
    assert kind in [
        "data_original",
        "data_train",
        "data_eval",
    ], f"ERROR! kind = {kind} unknown."

    if kind == "data_original":
        directory = env.data_original
    elif kind == "data_train":
        directory = env.data_train
    elif kind == "data_eval":
        directory = env.data_eval
    else:
        sys.exit("ERROR! should not occur.")

    return join(directory, f"{category}_{language}.jsonl")


def read_sampling_weights(
    percent: int = 100, verbose: bool = False
) -> Tuple[
    List[str], List[str], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]
]:
    """
    read sampling weights from SAMPLING_WEIGHTS.csv and weight them globally with 'percent'

    Args:
        percent: e.g. 50
        verbose: e.g. False

    Returns:
        categories: e.g. 'books', 'articles'
        languages: e.g. 'en'
        sampling_weights: e.g. {'books': {'en': 0.5}, 'articles': {'en': 1.0}}
        sampling_weights_final: e.g. {'books': {'en': 0.25}, 'articles': {'en': 0.5}}
    """
    categories = []
    languages = []
    sampling_weights = {}
    with open("SAMPLING_WEIGHTS.csv", "r", encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for j, row in enumerate(csv_reader):
            if j == 0:
                languages = row[1:]
            else:
                categories.append(row[0])
                sampling_weights[row[0]] = {
                    languages[i - 1]: float(row[i])
                    for i in range(1, len(languages) + 1)
                }

    sampling_weights_final = {
        _category: {_language: v2 * percent / 100 for _language, v2 in v1.items()}
        for _category, v1 in sampling_weights.items()
    }

    if verbose:
        print("\n> read sampling weights from SAMPLING_WEIGHTS.csv")
        print(f"  categories: {categories}")
        print(f"  languages: {languages}")
        print(f"  sampling_weights: {sampling_weights}")
        print(f"  sampling_weights_final: {sampling_weights_final}")

    return categories, languages, sampling_weights, sampling_weights_final


def reservoir_sampling(infile, number_of_sampled_documents: int):
    """
    taken from
    https://stackoverflow.com/questions/40144869/python-read-random-lines-from-a-very-big-file-and-append-to-another-file
    """
    iteration = iter(infile)
    try:
        result = [
            next(iteration) for _ in range(number_of_sampled_documents)
        ]  # use xrange if on python 2.x
    except StopIteration as exc:
        raise ValueError("Sample larger than population") from exc

    for i, item in enumerate(iteration, start=number_of_sampled_documents):
        random_int = random.randint(0, i)
        if random_int < number_of_sampled_documents:
            result[random_int] = item

    # random.shuffle(result)  # additional cost without effect
    return result
