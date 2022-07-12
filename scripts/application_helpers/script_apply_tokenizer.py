"""
EXECUTION: python script_apply_tokenizer.py
           --id 123456

PURPOSE: the script
         - loads the tokenizer with the given <id> (that needs to be present in the folder <output>/<id>_*)
         - applies it to the data in TEST_EXAMPLES and prints the result
"""
import argparse
import os
from os.path import join, isfile, isdir
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from os.path import abspath, dirname
import sys
BASE_DIR = abspath(dirname(dirname(dirname(abspath(__file__)))))
print(f">>> BASE_DIR: {BASE_DIR}")
sys.path.append(BASE_DIR)

from src.test_data import TEST_EXAMPLES
from src.env import Env


def main(args):
    env = Env()

    assert isdir(env.output), f"ERROR! output directory {env.output} does not exist."
    _ids = [elem for elem in os.listdir(env.output) if args.id in elem]
    assert len(_ids) == 1, f"ERRRO! _ids = {_ids} should be 1 element (env.output = {env.output})"
    _id = [elem for elem in os.listdir(env.output) if args.id in elem][0]
    tokenizer_file = join(env.output, _id, "tokenizer.json")
    assert isfile(tokenizer_file), f"ERROR! {tokenizer_file} does not exist (only HF implemented)."

    tokenizer = Tokenizer.from_file(tokenizer_file)
    # print(tokenizer.pre_tokenizer.pre_tokenize_str("Let's test pre-tokenization!"))

    # 1. Load BPE Tokenizer
    tokenizer_fast = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)

    # 2. Apply BPE Tokenizer
    for example in TEST_EXAMPLES:
        encoding = tokenizer_fast.encode(example)
        print("============")
        print(f"example: '{example}'")
        print(f"pre-tok: {tokenizer.pre_tokenizer.pre_tokenize_str(example)}")
        # print(encoding)
        print(f"encoded: {tokenizer_fast.convert_ids_to_tokens(encoding)}")
        print(f"decoded: '{tokenizer_fast.decode(encoding)}'")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, required=True)
    _args = parser.parse_args()

    main(_args)
