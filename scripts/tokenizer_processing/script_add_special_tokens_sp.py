"""
EXECUTION: python script_add_special_tokens_sp.py
           --tokenizer_directory <tokenizer_directory>

PURPOSE: the script
         - loads the tokenizer from <output>/<tokenizer_directory>/model.model
         - adds a special token
         - writes the new tokenizer to <output>/<tokenizer_directory>___MST/model.model
"""

from os.path import join

from os.path import abspath, dirname
import sys
BASE_DIR = abspath(dirname(dirname(dirname(abspath(__file__)))))
print(f">>> BASE_DIR: {BASE_DIR}")
sys.path.append(BASE_DIR)

import argparse
from src.helpers import add_special_tokens
from src.env import Env


def main(_args):
    env = Env()

    model_path = join(env.output, _args.tokenizer_directory)
    add_special_tokens(model_path, overwrite=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_directory", type=str, required=True)
    _args = parser.parse_args()

    main(_args)

