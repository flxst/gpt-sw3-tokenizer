"""
EXECUTION: python script_sampling_disjunct.py

PURPOSE: the script
     - reads the SAMPLING.json file in <data_train> & <data_eval>
     - checks whether the indices for each combination <category> & <language> are disjunct
"""
from src.env import Env
from os.path import join, isfile
from itertools import product
from src.helpers import read_json


def main():
    env = Env()
    categories, languages, _, _ = env.read_sampling_weights()

    # read SAMPLING.json for data_train & data_eval
    path_sampling = {
        "train": join(env.data_train, "SAMPLING.json"),
        "eval": join(env.data_eval, "SAMPLING.json"),
    }
    for key in path_sampling:
        assert isfile(path_sampling[key]), f"ERROR! file = {path_sampling[key]} does not exist."

    sampling = {
        key: read_json(path_sampling[key])
        for key in path_sampling.keys()
    }

    # check that all keys are present
    assert sampling["train"].keys() == sampling["eval"].keys(), \
        f"ERROR! keys are not the same for train ({sampling['train'].keys()} and eval ({sampling['eval'].keys()}"

    for category, language in product(categories, languages):
        category_language = f"{category}_{language}.jsonl"
        for key in sampling.keys():
            assert category_language in sampling[key], f"ERROR! {category_language} not in sampling[{key}]."

    # make sure that train & eval indices are disjunct for all categories & languages
    for category, language in product(categories, languages):
        category_language = f"{category}_{language}.jsonl"
        indices_train = sampling["train"][category_language]
        indices_eval = sampling["eval"][category_language]
        indices_intersection = set(indices_train).intersection(indices_eval)
        assert len(indices_intersection) == 0, \
            f"ERROR! train & eval indices are not disjunct for key = {category_language}: {indices_intersection}"

    print("> script_sampling_disjunct successful.")


if __name__ == "__main__":
    main()
