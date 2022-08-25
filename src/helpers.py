from tokenizers import normalizers
from datasets import Dataset
import os
import json
from typing import List
import time
import shutil
from os.path import join
from sentencepiece import sentencepiece_model_pb2 as model_pb2

LANGUAGES = ["cd", "da", "en", "is", "no", "sv"]


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
                       verbose: bool = False) -> None:
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


def create_merge_rules(_vocab_file: str, _merge_file: str, verbose: bool = False) -> List[str]:

    print("\n=== CREATE MERGE RULES ===")

    # 1. read vocabulary file
    with open(_vocab_file, "r", encoding="utf-8") as file:
        r = json.load(file)
    vocab = list(r.keys())
    print()
    print(f"> found {len(vocab)} subwords in {_vocab_file}")

    # 2a. find index of first "real" subword (i.e. no special token, byte fallback, 1-char-token)
    for i, subword in enumerate(vocab):
        if subword.startswith("<") and subword.endswith(">"):  # byte fallback (or special token)
            continue
        elif list(set(subword)) == ["▁"] or list(set(subword)) == [" "]:  # special whitespace token
            continue
        elif len(subword) == 1:  # one character
            continue
        else:
            idx_a = i
            break
    print(f"> the first {idx_a} subwords are special tokens, byte fallback tokens or 1-char-tokens")

    # 2b. find index of first non-one-character, non-whitespace subword (starting from the end)
    for i, subword in enumerate(reversed(vocab)):
        if len(subword) == 1:  # one-character subword
            continue
        elif list(set(subword)) == ["▁"] or list(set(subword)) == [" "]:  # special whitespace token
            continue
        else:
            idx_b = i
            break
    print(f"> the last  {idx_b} subwords are special whitespace tokens or 1-char-tokens")

    # 2c. find index of first non-whitespace subword (starting from the end)
    for i, subword in enumerate(reversed(vocab)):
        if list(set(subword)) == ["▁"] or list(set(subword)) == [" "]:  # special whitespace token
            continue
        else:
            idx_c = i
            break
    print(f"> the last  {idx_c} subwords are special whitespace tokens")

    # 3. create merge rules
    ts = time.time()
    _merge_rules = list()
    vocab_whitespace = vocab[-idx_c:] if idx_c > 0 else []
    vocab_filtered = vocab[:-idx_b][idx_a:] if idx_b > 0 else vocab[idx_a:]
    vocab_filtered += vocab_whitespace
    assert len(vocab_filtered) == len(vocab) - idx_a - idx_b + idx_c, \
        f"ERROR! len(vocab_filtered) = {len(vocab_filtered)}, " \
        f"len(vocab) = {len(vocab)}, idx_a = {idx_a}, idx_b = {idx_b}, idx_c = {idx_c}"

    print()
    print(f"> find merge rules for {len(vocab_filtered)} = {len(vocab)} - {idx_a} - {idx_b} + {idx_c} subwords")

    def _get_index(_list: List[str], _subword: str) -> int:
        if _subword in _list:
            return _list.index(_subword)
        else:
            return -1

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
            _previous_vocab = dict()  # maps vocab index to subword, e.g. {2: 'de'}
            if verbose:
                print()
                print("========")
                print(f"i, subword = {i}, {subword}")
            for idx_start in range(0, len(subword)):
                for idx_end in range(idx_start + 1, len(subword) + 1):
                    chunk = subword[idx_start: idx_end]
                    _idx = _get_index(vocab_filtered[:i], chunk)
                    if _idx > -1:
                        _previous_vocab[_idx] = chunk  # _previous_vocab[_idx]
                    error = 0

            if error:
                merge_rule = "---"
                error_counter += 1
            else:
                _list = [char for char in subword]
                _previous_vocab = {k: v for (k, v) in sorted(_previous_vocab.items())}
                if verbose:
                    print("_list:", _list)
                    print("_previous_vocab:", _previous_vocab)

                while 1:
                    llist_before = len(_list)
                    for _, v in _previous_vocab.items():
                        adj = 0
                        len_list = len(_list)
                        for j in range(len(_list) - 1):
                            if j >= len_list - 1 - adj:
                                break
                            for w in range(1, len(v)):
                                subword_1 = v[:w]
                                subword_2 = v[w:]
                                if _list[j] == subword_1 and _list[j+1] == subword_2:
                                    _list[j] = "".join(_list[j: j+2])
                                    del _list[j+1]
                                    adj += 1
                                    break
                        if len(_list) == 2:
                            break
                    llist_after = len(_list)
                    assert llist_after < llist_before, \
                        f"ERROR! length of list did not decrase! " \
                        f"llist_before = {llist_before}, llist_after = {llist_after}, list = {_list}"
                    if len(_list) == 2:
                        break

                if verbose:
                    print("_list:", _list)

                assert len(_list) <= 2, f"ERROR! len(_list) = {len(_list)}"

                merge_rule = f"{_list[0]} {_list[1]}"

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
