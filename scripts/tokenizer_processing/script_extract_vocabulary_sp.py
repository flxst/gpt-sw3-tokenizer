"""
EXECUTION: python script_extract_vocabulary_sp.py
           --tokenizer_directory <tokenizer_directory>

PURPOSE: the script
         - loads the tokenizer from <output>/<tokenizer_directory>/model.model
         - extracts the vocabulary and writes it to <output>/<tokenizer_directory>/tokenizer_vocab.json
         this is also done by script_evaluate.py
"""
import sys
from os.path import join, abspath, dirname
import argparse

BASE_DIR = abspath(dirname(dirname(dirname(abspath(__file__)))))
print(f">>> BASE_DIR: {BASE_DIR}")
sys.path.append(BASE_DIR)

from src.env import Env
from src.analysis import extract_vocab


def main(_args):
    env = Env()
    model = join(env.output, _args.tokenizer_directory, "model.model")
    extract_vocab(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_directory", type=str, required=True)
    _args = parser.parse_args()

    main(_args)


