"""Module that contains functions for the analysis of a tokenizer's vocabulary"""
from os.path import join, isfile
import json
import re
from typing import Dict, List

# from sentencepiece import sentencepiece_model_pb2 as model
import sentencepiece as spm
from src.env import Env


def _analyze_vocab(_env: Env, _model: str) -> Dict[str, List[int]]:
    """
    analyze vocabulary of tokenizer '_model'

    Args:
        _env: environment
        _model: e.g. <output>/151508_SP-uNone-d0-p0-w0-c0-f0-bf0-cc1.0-x1-v1000_SP_test

    Returns:
        indices: e.g. {
            'special': [0, 1, 2, 3],
            '<|*|>': [4, 5, 6, 7],
            [..]
        }
    """
    vocab_file = join(_model, "tokenizer_vocab.json")
    vocab_file_full = join(_env.output, vocab_file)
    assert isfile(
        vocab_file_full
    ), f"ERROR! vocab_file = {vocab_file_full} does not exist."
    with open(vocab_file_full, "r", encoding="utf-8") as file:
        vocab_dict = json.load(file)
    print(f"> read vocabulary of size = {len(vocab_dict)} \n  from {vocab_file}")

    vocab = list(vocab_dict.keys())

    indices: Dict[str, List[int]] = {
        key: []
        for key in [
            "special",
            "<|*|>",
            "<0x*>",
            "merges",
            "pruned",
            "single_character",
            "whitespace",
        ]
    }

    # 1. special token
    _pad = [i for i, item in enumerate(vocab) if re.search("<pad>", item)]
    _unk = [i for i, item in enumerate(vocab) if re.search("<unk>", item)]
    _s = [i for i, item in enumerate(vocab) if re.search("<s>", item)]
    _eos = [i for i, item in enumerate(vocab) if re.search(r"<\|endoftext\|>", item)]
    indices["special"] = _pad + _unk + _s + _eos

    # 2. <|*|> tokens
    indices["<|*|>"] = [
        i
        for i, item in enumerate(vocab)
        if re.search(r"<\|.+\|>", item) and item != "<|endoftext|>"
    ]

    # 3. <0x*> byte fallback tokens
    indices["<0x*>"] = [i for i, item in enumerate(vocab) if re.search("<0x.+>", item)]

    # X. pruned tokens
    indices["pruned"] = [
        i
        for i, item in enumerate(vocab)
        if re.search(r"a\!\?x\$\$\▁\!\!xyz\.masdf", item)
    ]

    # 5. single character tokens
    indices["single_character"] = [i for i, item in enumerate(vocab) if len(item) == 1]

    # 6. whitespace tokens
    indices["whitespace"] = [
        i for i, item in enumerate(vocab) if len(item) > 1 and list(set(item)) == ["▁"]
    ]

    # 4. merges
    all_indices = []
    for key in indices.keys():
        all_indices.extend(indices[key])
    indices["merges"] = [i for i in range(len(vocab)) if i not in all_indices]

    extracted_indices = sum(len(v) for v in indices.values())
    assert extracted_indices == len(
        vocab
    ), f"ERROR! extracted {extracted_indices} from {len(vocab)} indices"

    def stringify(_list):
        if len(_list) == 0:
            return "---"
        if len(_list) == 1:
            return f"{_list[0]} (#={len(_list)})"
        if len(_list) == _list[-1] - _list[0] + 1:
            return f"{_list[0]}-{_list[-1]} (#={len(_list)})"
        return "ERROR!!"

    indices_str = {key: stringify(indices[key]) for key in indices.keys()}

    print()
    print("=== overview vocabulary ===")
    for key, value in indices_str.items():
        print(f"{key}: {value}")

    return indices


def extract_vocab(_model: str) -> None:
    """
    extract vocabulary from tokenizer

    Args:
        _model: path to tokenizer, e.g. [..]/model.model

    Output:
        [..]/tokenizer_vocab.json
    """
    spp = spm.SentencePieceProcessor()
    spp.Load(_model)
    vocabs = {spp.IdToPiece(_id): _id for _id in range(spp.GetPieceSize())}

    env = Env()
    if env.debug:
        print("\n===")
        print(f"vocabulary size = {len(vocabs)}")
        print()
        print("examples:")
        for _id in range(5):
            print(spp.IdToPiece(_id), vocabs[spp.IdToPiece(_id)])
        for _id in range(len(vocabs) - 5, len(vocabs)):
            print(spp.IdToPiece(_id), vocabs[spp.IdToPiece(_id)])
        print()

    # m = model.ModelProto()
    # m.ParseFromString(open(checkpoint, 'rb').read())
    # print(m.pieces)

    vocab_file = _model.replace("model.model", "tokenizer_vocab.json")
    if isfile(vocab_file):
        print(f"> vocabulary file {vocab_file} already exists.")
        print("SKIPPED.")
    else:
        print(f"> write vocabulary to {vocab_file}")
        with open(vocab_file, "w", encoding="utf-8") as file:
            json.dump(vocabs, file)
        print("DONE.")
