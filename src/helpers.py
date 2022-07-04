from tokenizers import normalizers
from datasets import Dataset
import os
import json
from typing import List
import time
import shutil
from os.path import join
from sentencepiece import sentencepiece_model_pb2 as model_pb2

UNICODE_NORMALIZATION = {
    "None": None,
    "NFC": normalizers.NFC(),
    "NFKC": normalizers.NFKC(),
    "NFKD": normalizers.NFKD(),
}

LIST_OF_SPECIAL_TOKENS = [
    "▁" * i for i in range(2, 25)
]


def get_normalizer(_unicode_normalization: str):
    return UNICODE_NORMALIZATION[_unicode_normalization]


def get_training_corpus_combined(_dataset: Dataset, batch_size: int = 100000):
    for i in range(0, len(_dataset['train']), batch_size):
        yield str(_dataset['train'][i: i + batch_size]["text"])


def add_special_tokens(_model_path: str,
                       overwrite: bool = True,
                       verbose: bool = False):
    print("\n=== ADD SPECIAL TOKENS ===")

    _vocab_path = join(_model_path, "model.vocab")
    if overwrite:
        _new_model_path = _model_path
        _new_vocab_path = _vocab_path
    else:
        _new_model_path = _model_path + "___MST"
        _new_vocab_path = join(_new_model_path, "model.vocab")

    # A1. read model
    if verbose:
        print("\n--- 1. read model ---")
    m = model_pb2.ModelProto()
    m.ParseFromString(open(join(_model_path, 'model.model'), 'rb').read())
    lowest_score = m.pieces[-1].score
    if verbose:
        print(f"> lowest_score = {lowest_score}")

    # A2. unprioritize accidental special tokens (special tokens that happen to exist as learned pieces)
    if verbose:
        print("\n--- 2. unprioritize accidental special tokens ---")
    counter_unprioritize = 0
    for p in m.pieces:
        if list(set(p.piece)) == ["▁"] and p.piece in LIST_OF_SPECIAL_TOKENS:
            if verbose:
                print(f"> unprioritize {p.piece}")
            p.piece = f"UNPRIORITIZED_{counter_unprioritize}"
            p.score = lowest_score - 2
            counter_unprioritize += 1
    if verbose:
        print(f"> unprioritized {counter_unprioritize} accidental special tokens")

    # A3. add special tokens
    if verbose:
        print("\n--- 3. add special tokens ---")
        print(len(m.pieces))
        print(m.pieces[-1])
    for special_token in LIST_OF_SPECIAL_TOKENS:
        m.pieces.add()
        m.pieces[-1].piece = special_token
        m.pieces[-1].score = lowest_score - 1
    if verbose:
        print(f"> added {len(LIST_OF_SPECIAL_TOKENS)} pieces with score = {lowest_score - 1}")

    # A4. write new model
    if verbose:
        print("\n--- 4. write new model ---")
    os.makedirs(_new_model_path, exist_ok=True)
    with open(join(_new_model_path, 'model.model'), 'wb') as f:
        f.write(m.SerializeToString())
    if verbose:
        print(f"> wrote new model to {join(_new_model_path, 'model.model')}")

    # B1. add special tokens to vocab file
    if not overwrite:
        shutil.copyfile(_vocab_path, _new_vocab_path)

    with open(_new_vocab_path, "a") as f:
        for special_token in LIST_OF_SPECIAL_TOKENS:
            f.write(f"{special_token}\t{int(lowest_score-1)}\n")


def create_merge_rules(_vocab_file: str, _merge_file: str) -> List[str]:

    print("\n=== CREATE MERGE RULES ===")

    # 1. read vocabulary file
    with open(_vocab_file, "r", encoding="utf-8") as file:
        r = json.load(file)
    vocab = list(r.keys())
    print()
    print(f"> found {len(vocab)} subwords in {_vocab_file}")

    # 2a. find index of first "real" subword (i.e. no special token or byte fallback)
    for i, subword in enumerate(vocab):
        if subword.startswith("<") and subword.endswith(">"):  # byte fallback (or special token)
            continue
        elif list(set(subword)) == ["▁"]:  # special whitespace token
            continue
        else:
            idx_a = i
            break
    print(f"> the first {idx_a} subwords are special tokens or byte fallback tokens")

    # 2b. find index of first non-one-character subword (starting from the end)
    for i, subword in enumerate(reversed(vocab)):
        if len(subword) == 1:  # one-character subword
            continue
        elif list(set(subword)) == ["▁"]:  # special whitespace token
            continue
        else:
            idx_b = i
            break
    print(f"> the last  {idx_b} subwords are one-character tokens")

    # 3. create merge rules
    ts = time.time()
    _merge_rules = list()
    vocab_filtered = vocab[:-idx_b][idx_a:]
    assert len(vocab_filtered) == len(vocab) - idx_a - idx_b, \
        f"ERROR! len(vocab_filtered) = {len(vocab_filtered)}, len(vocab) = {len(vocab)}, idx_a = {idx_a}, idx_b = {idx_b}"

    print()
    print(f"> find merge rules for {len(vocab_filtered)} = {len(vocab)} - {idx_a} - {idx_b} subwords")

    error_counter = 0
    for i, subword in enumerate(vocab_filtered):
        if i % 10000 == 0:
            print(f"... i = {i}")

        if len(subword) == 1:
            print(f"ERROR! found subword with len == 1: i={i}, subword = {subword}, repr(subword) = {repr(subword)}")
            exit()
        elif len(subword) == 2:
            subword_1 = subword[:1]
            subword_2 = subword[1:]
            merge_rule = f"{subword_1} {subword_2}"
        elif len(subword) > 2:
            error = 1
            for len_first_subword in range(1, len(subword)):
                subword_1 = subword[:len_first_subword]
                subword_2 = subword[len_first_subword:]
                if subword_1 in vocab_filtered[:i] or subword_2 in vocab_filtered[:i]:
                    merge_rule = f"{subword_1} {subword_2}"
                    error = 0
                    break

            if error:
                merge_rule = "---"
                error_counter += 1
        _merge_rules.append(merge_rule)

    if 1:
        print(f"> found {len(_merge_rules)} merge rules, {error_counter} errors")
        assert len(_merge_rules) == len(vocab_filtered), \
            f"ERROR! len(merge_rules) = {len(_merge_rules)} != len(vocab_filtered) = {len(vocab_filtered)}"
    if 0:
        for a, b in zip(merge_rule, vocab_filtered):
            print(f"{a} --> {b}")

    # 4. write merge rule file
    with open(_merge_file, "w", encoding="utf-8") as f:
        for merge_rule in _merge_rules:
            f.write(merge_rule + "\n")
    print(f"> wrote merges file '{_merge_file}': #merges = {len(_merge_rules)}")

    # 5. end
    te = time.time()
    print(f"\n>>> time = {te-ts:.1f}s")

    return _merge_rules


