"""
EXECUTION: python script_create_test_data.py

PURPOSE: the script creates data files for testing:
         - <data_sampled>/test.json   (contains TEST_CORPUS)
         - <data_sampled>/code.json   (contains script_train.py as string)
         - <data_sampled>/fibrec.json (contains fibRec function as string)
"""
import os
from os.path import join
import json
from src.test_data import TEST_CORPUS
from src.env import Env


def _add_features(_d):
    features = {
        "title": "",
        "filename": "",
        "filters": [""],
        "keep": 1,
        "len_char": 0,
        "len_utf8bytes": 0,
        "len_words": 0,
        "len_sents": 0,
        "lang": "",
        "md5": "",
    }
    return {**_d, **features}


def main():
    env = Env()
    os.makedirs(env.data_sampled, exist_ok=True)

    test_data_file = join(env.data_sampled, "test.json")
    with open(test_data_file, "w", encoding="utf-8") as f:
        for i in range(len(TEST_CORPUS)):
            f.write(json.dumps({"text": TEST_CORPUS[i]}) + "\n")
    print(f"> wrote file '{test_data_file}'")

    with open("script_train.py", "r") as f:
        code = f.read().replace("\\n", "\n")
    print(code)

    code_data_file = join(env.data_sampled, "code.json")
    _dict = {"text": code}
    _dict = _add_features(_dict)
    with open(code_data_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(_dict))
    print(f"> wrote file '{code_data_file}'")

    fib_rec_file = join(env.data_sampled, "fibrec.json")
    fib_rec = "def fibRec(n):\n    if n < 2:\n        return n\n    else:\n        return fibRec(n-1) + fibRec(n-2)"
    _dict = {"text": fib_rec}
    _dict = _add_features(_dict)
    with open(fib_rec_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(_dict))
    print(f"> wrote file '{fib_rec_file}'")


if __name__ == "__main__":
    main()