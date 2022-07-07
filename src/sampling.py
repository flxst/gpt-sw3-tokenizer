
import csv
from typing import Tuple, List, Dict
from os.path import join
from src.env import Env

env = Env()


def get_file_path(category: str, language: str, original: bool = True, percent: int = 0) -> str:
    if original:
        return join(env.data_original, f"{category}_{language}.jsonl")
    else:
        return join(env.data_sampled, f"{category}_{language}_{percent}p.jsonl")


def read_sampling_weights(percent: int = 100,
                          verbose: bool = False) -> Tuple[List[str],
                                                          List[str],
                                                          Dict[str, Dict[str, float]],
                                                          Dict[str, Dict[str, float]]]:
    categories = list()
    languages = list()
    sampling_weights = dict()
    with open("SAMPLING_WEIGHTS.csv", "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for r, row in enumerate(csv_reader):
            if r == 0:
                languages = row[1:]
            else:
                categories.append(row[0])
                sampling_weights[row[0]] = {languages[i-1]: float(row[i]) for i in range(1, len(languages)+1)}

    sampling_weights_final = {
        _category: {
            _language: v2*percent/100 for _language, v2 in v1.items()
        }
        for _category, v1 in sampling_weights.items()
    }

    if verbose:
        print(f"\n> read sampling weights from SAMPLING_WEIGHTS.csv")
        print(f"  categories: {categories}")
        print(f"  languages: {languages}")
        print(f"  sampling_weights: {sampling_weights}")
        print(f"  sampling_weights_final: {sampling_weights_final}")

    return categories, languages, sampling_weights, sampling_weights_final
