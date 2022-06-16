import argparse
import os
from os.path import join, isfile, isdir
from tokenizers import Tokenizer
from test_data import TEST_EXAMPLES
from transformers import PreTrainedTokenizerFast


def main(args):
    assert isdir("output"), f"ERROR! output directory does not exist."
    _id = [elem for elem in os.listdir("output") if args.id in elem][0]
    tokenizer_file = join("output", _id, "tokenizer.json")
    assert isfile(tokenizer_file), f"ERROR! {tokenizer_file} does not exist."

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
