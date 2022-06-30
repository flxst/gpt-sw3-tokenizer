
from os.path import join
import json
import time
from typing import List


def merge(_vocab_file: str) -> List[str]:

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

    te = time.time()
    print(f"\n>>> time = {te-ts:.1f}s")

    return _merge_rules


if __name__ == "__main__":
    model_name = "093017_SP-uNone-d1-p1-w1-c1-f0-bf1-cc0.9999-x1-v128000_3da"
    vocab_file = join("output", model_name, "tokenizer_vocab.json")

    merge_rules = merge(vocab_file)

    merge_file = join("output", model_name, "tokenizer_merge.txt")
    with open(merge_file, "w", encoding="utf-8") as f:
        for merge_rule in merge_rules:
            f.write(merge_rule + "\n")
    print(f"> wrote merges file '{merge_file}': #merges = {len(merge_rules)}")
