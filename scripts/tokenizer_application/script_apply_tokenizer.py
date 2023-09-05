"""
EXECUTION: python script_apply_tokenizer.py
           --tokenizer_directory <tokenizer_directory>
           [--SP]
           [--HF]

PURPOSE: the script
         - loads the tokenizer from <output>/<tokenizer_directory>/tokenizer.json or [..]/model.model
         - applies it to the data in TEST_EXAMPLES and prints the result
         - the same is done by script_evaluate.py
"""
import argparse
import sentencepiece as spm
from os.path import join, isfile, isdir
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from os.path import abspath, dirname
import sys
BASE_DIR = abspath(dirname(dirname(dirname(abspath(__file__)))))
print(f">>> BASE_DIR: {BASE_DIR}")
sys.path.append(BASE_DIR)

from src.hardcoded.test_data import TEST_EXAMPLES
from src.env import Env

NOT_AVAILABLE_STR = "---"


def id2piece(_tokenizer, _id):
    try:
        piece = _tokenizer.IdToPiece(_id)
    except IndexError:
        piece = NOT_AVAILABLE_STR
    return piece


def main(args):
    env = Env()

    tokenizer_type = "HF" if args.HF else "SP"

    assert isdir(env.output), f"ERROR! output directory {env.output} does not exist."

    if tokenizer_type == "HF":
        tokenizer_file = join(env.output, args.tokenizer_directory, "tokenizer.json")
        assert isfile(tokenizer_file), f"ERROR! {tokenizer_file} does not exist."

        tokenizer = Tokenizer.from_file(tokenizer_file)
        # print(tokenizer.pre_tokenizer.pre_tokenize_str("Let's test pre-tokenization!"))

        # 1. Load BPE Tokenizer
        tokenizer_fast = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
    else:  # SP
        tokenizer_file = join(env.output, args.tokenizer_directory, "model.model")  # TODO: temp
        assert isfile(tokenizer_file), f"ERROR! {tokenizer_file} does not exist."

        # 1. Load BPE Tokenizer
        tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_file)

    # 2. Apply BPE Tokenizer
    for example in TEST_EXAMPLES:
        if tokenizer_type == "HF":
            encoding = tokenizer_fast.encode(example)
            print("============")
            print(f"example: '{example}'")
            print(f"pre-tok: {tokenizer.pre_tokenizer.pre_tokenize_str(example)}")
            # print(encoding)
            print(f"encoded: {tokenizer_fast.convert_ids_to_tokens(encoding)}")
            print(f"decoded: '{tokenizer_fast.decode(encoding)}'")
            print()
        else:  # SP
            encoding = tokenizer.encode(example)
            print("============")
            print(f"example: '{example}'")
            print(f"encoded: {encoding}")
            print(f"decoded: '{tokenizer.decode(encoding)}'")
            print()
            assert tokenizer.decode(tokenizer.encode(example)) == example, f"ERROR!"

    if tokenizer_type == "SP":
        pad_piece = id2piece(tokenizer, tokenizer.pad_id())
        unk_piece = id2piece(tokenizer, tokenizer.unk_id())
        bos_piece = id2piece(tokenizer, tokenizer.bos_id())
        eos_piece = id2piece(tokenizer, tokenizer.eos_id())

        print("--- SENTENCE PIECE SPECIAL TOKENS ---")
        print(f"tokenizer.pad_id() = {tokenizer.pad_id()} --> IdToPiece = {pad_piece}")
        print(f"tokenizer.unk_id() = {tokenizer.unk_id()} --> IdToPiece = {unk_piece}")
        print(f"tokenizer.bos_id() = {tokenizer.bos_id()} --> IdToPiece = {bos_piece}")
        print(f"tokenizer.eos_id() = {tokenizer.eos_id()} --> IdToPiece = {eos_piece}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_directory", type=str, required=True)
    parser.add_argument("--SP", action="store_true")
    parser.add_argument("--HF", action="store_true")
    _args = parser.parse_args()
    assert _args.SP is True or _args.HF is True, f"ERROR! Need to specify either '--sp' or '--hf'"

    main(_args)
