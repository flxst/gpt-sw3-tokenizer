
from os.path import join, isfile
import json
import re

from src.env import Env


def analyze_vocab(_model):
    env = Env("..")

    vocab_file = join(_model, "tokenizer_vocab.json")
    vocab_file_full = join(env.output, vocab_file)
    assert isfile(vocab_file_full), f"ERROR! vocab_file = {vocab_file_full} does not exist."
    with open(vocab_file_full, "r") as f:
        vocab_dict = json.load(f)
    print(f"> read vocabulary of size = {len(vocab_dict)} \n  from {vocab_file}")

    vocab = list(vocab_dict.keys())

    indices = {
        key: list()
        for key in ["<unk>", "<|*|>", "<0x*>", "merges", "single_character", "whitespace"]
    }

    # 1. <unk> token
    indices["<unk>"] = [i for i, item in enumerate(vocab) if re.search('<unk>', item)]

    # 2. <|*|> special tokens
    indices["<|*|>"] = [i for i, item in enumerate(vocab) if re.search('<\|.+\|>', item)]

    # 3. <0x*> byte fallback tokens
    indices["<0x*>"] = [i for i, item in enumerate(vocab) if re.search('<0x.+>', item)]

    # 5. single character tokens
    indices["single_character"] = [i for i, item in enumerate(vocab) if len(item) == 1]

    # 6. whitespace tokens
    indices["whitespace"] = [i for i, item in enumerate(vocab) if len(item) > 1 and list(set(item)) == ['▁']]

    # print(indices)

    # 4. merges
    all_indices = list()
    for key in indices.keys():
        all_indices.extend(indices[key])
    indices["merges"] = [i for i in range(len(vocab)) if i not in all_indices]

    extracted_indices = sum([len(v) for v in indices.values()])
    assert extracted_indices == len(vocab), f"ERROR! extracted {extracted_indices} from {len(vocab)} indices"

    def stringify(_list):
        if len(_list) == 1:
            return f"{_list[0]} (#={len(_list)})"
        elif len(_list) == _list[-1] - _list[0] + 1:
            return f"{_list[0]}-{_list[-1]} (#={len(_list)})"
        else:
            return "ERROR!!"

    indices_str = {
        key: stringify(indices[key])
        for key in indices.keys()
    }

    print()
    print("=== overview vocabulary ===")
    for k, v in indices_str.items():
        print(f"{k}: {v}")
