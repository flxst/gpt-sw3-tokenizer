
import argparse
import os
from os.path import isdir, join

from os.path import abspath, dirname
import sys
BASE_DIR = abspath(dirname(dirname(dirname(abspath(__file__)))))
print(f">>> BASE_DIR: {BASE_DIR}")
sys.path.append(BASE_DIR)

from src.env import Env


def main(args):
    env = Env()
    assert isdir(env.output), f"ERROR! directory = {env.output} does not exist"
    files = [elem for elem in os.listdir(env.output) if isdir(join(env.output, elem))]
    files_filtered = [file for file in files if f"-v{args.vocab_size}_{args.tokenizer_number}" in file]

    for file in files_filtered:
        old_file = join(env.output, file)
        new_file = join(env.output, "multilinguality", file)
        os.rename(old_file, new_file)
        print(f"> {old_file} moved to {new_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_number", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    _args = parser.parse_args()

    main(_args)