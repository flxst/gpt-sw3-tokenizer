"""
EXECUTION: python script_evaluate.py
           [parameters = <tokenizers>, <datasets> are hardcoded in the script]

PURPOSE: the script
         - applies each tokenizer on each dataset and computes unk_rate & closeness_to_character_level
         - writes results to `<OUTPUT>/evaluation/results_*.json`
"""

from copy import deepcopy
import os
from os.path import isfile, join, isdir
import json
import argparse
# from transformers import PreTrainedTokenizerFast
import sentencepiece as spm
from collections import Counter
import time
from itertools import product
from typing import Tuple, List
from sentencepiece import sentencepiece_model_pb2 as model_pb2
from src.env import Env

env = Env()
DATA_DIR = env.data_sampled
OUTPUT_DIR = env.output
DEBUG = 0
VERBOSE = 0


def evaluate(_model_dir, _data_path):
    assert isdir(_model_dir), f"ERROR! model_dir = {_model_dir} does not exist."
    ts = time.time()

    # 0. load data
    with open(_data_path, "r", encoding="utf-8") as file:
        _data = [json.loads(line)["text"] for line in file]

    _unk_rate = None
    _closeness_to_character_level = None

    # 1. tokenize data
    if isfile(join(_model_dir, "tokenizer.json")):
        library = "HF"
        print("HF not implemented.")
        """
        tokenizer_file = join(_model_dir, "tokenizer.json")
        tokenizer_fast = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
        """
    elif isfile(join(_model_dir, "model.model")):
        library = "SP"
        tokenizer_file = join(_model_dir, "model.model")
        sp = spm.SentencePieceProcessor(model_file=tokenizer_file)

        number_of_unk = 0
        sentence_length_in_subwords = 0
        sentence_length_in_characters = 0
        for example in _data:
            sentence_length_in_characters += len(example)

            encoding = sp.encode(example, out_type=int)
            sentence_length_in_subwords += len(encoding)

            counter = Counter(encoding)
            number_of_unk += counter[0]

            # example_encoded = sp.encode(example, out_type=str)
            # assert len(example_encoded) == len(encoding)

            if DEBUG:
                print(example[:50])
                print(encoding[:10])
                print(counter[0])
                # print(example_encoded[:10])
                break

        try:
            _unk_rate = number_of_unk/float(sentence_length_in_subwords)
        except ZeroDivisionError:
            _unk_rate = -1

        try:
            _closeness_to_character_level = float(sentence_length_in_subwords)/sentence_length_in_characters
        except ZeroDivisionError:
            _closeness_to_character_level = -1

        if VERBOSE:
            print()
            print(f"[time = {time.time() - ts:.2f}s]")
            print()
            print(f"> number_of_unk = {number_of_unk}")
            print(f"> sentence_length_in_subwords = {sentence_length_in_subwords}")
            print(f"> sentence_length_in_characters = {sentence_length_in_characters}")
            print()
            print(f"=> unk rate = {_unk_rate:.3f}")
            print(f"=> closeness to character level = {_closeness_to_character_level:.3f}")

    return _unk_rate, _closeness_to_character_level


def extract_bf_cc_from_model(_model) -> Tuple[str, str]:
    _bf = _model.split("-bf")[1].split("-cc")[0]
    _cc = _model.split("-cc")[1].split("-v")[0]
    return _bf, _cc


def get_models(_name: str) -> List[str]:
    subdirs = [elem for elem in os.listdir(env.output) if isdir(join(env.output, elem)) and elem.endswith(_name)]
    assert len(subdirs) > 0, f"ERROR! did not find any subdirectories that end with {_name} in env.output = {env.output}"
    assert len(subdirs) == 1, f"ERROR! found multiple subdirectories: {subdirs}"
    _models = subdirs
    return _models


def prune_vocab_size(_models: str,
                     _vocab_sizes: List[int]) -> List[str]:
    """only works for library == SP"""
    _new_models = [_models]
    vocab_size_model = _vocab_sizes[-1]
    assert str(vocab_size_model) in _models, \
        f"ERROR! vocab size = {vocab_size_model} is not in _models = {_models}"
    model_file = join(_models, "model.model")
    m = model_pb2.ModelProto()
    m.ParseFromString(open(model_file, 'rb').read())
    for _vocab_size in _vocab_sizes[:-1]:
        pruned_model_dir = _models.replace(f"v{vocab_size_model}", f"v{_vocab_size}")
        os.makedirs(pruned_model_dir, exist_ok=False)
        pruned_model = join(pruned_model_dir, "model.model")

        m_pruned = deepcopy(m)
        for i, _ in enumerate(m.pieces):
            if _vocab_size - 23 < i < vocab_size_model - 23:  # TODO: this works only for add_whitespace_tokens == 2
                # workaround: overwrite with extremely unlikely token
                m_pruned.pieces[i].piece = f"a!?x$$▁!!xyz.masdf_{i}"

        for j in [0, 1, 2, _vocab_size-25, _vocab_size-24]:
            assert m.pieces[j].piece == m_pruned.pieces[j].piece, \
                f"ERROR for j = {j}, piece: {m.pieces[j].piece} != {m_pruned.pieces[j].piece}"
            assert m.pieces[j].score == m_pruned.pieces[j].score, \
                f"ERROR for j = {j}, score: {m.pieces[j].score} != {m_pruned.pieces[j].score}"

        if DEBUG:
            for j in [0, _vocab_size - 23, _vocab_size - 22, -24, -23, -1]:
                print(j)
                print(m.pieces[j])
                print(m_pruned.pieces[j])

        with open(pruned_model, 'wb') as f:
            f.write(m_pruned.SerializeToString())
        print(f"> wrote new model to {pruned_model}")

        _new_models.append(pruned_model_dir)

    return _new_models


def main(_name, _models, _vocab_sizes=None):
    _models = [join(env.output, model) for model in _models]
    _data_eval = [join(env.data_eval, elem) for elem in os.listdir(env.data_eval)]

    if len(_models) == 1:
        _models = prune_vocab_size(_models[0], _vocab_sizes)  # returns list

    results = {
        model: {
            data: dict()
            for data in _data_eval
        }
        for model in _models
    }

    for model, data in product(_models, _data_eval):
        # bf, cc = extract_bf_cc_from_model(model)
        if DEBUG:
            print()
            print(f"> model = {model}")
            print(f"> data = {data}")
        Xunk_rate, Xcloseness_to_character_level = evaluate(model, data)
        results[model][data]["unk_rate"] = Xunk_rate
        results[model][data]["closeness_to_character_level"] = Xcloseness_to_character_level

    print()
    print("--- results ---")
    print(results)
    print("---------------")

    os.makedirs(join(env.output, "evaluation"), exist_ok=True)
    results_file = join(env.output, "evaluation", f"results_{_name}.json")
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(results))
    print(f"> wrote results to {results_file}")


if __name__ == "__main__":
    if 0:
        name = "bf-cc-test"
        models = [
            "194127_SP-uNone-d1-p1-w1-c1-f0-bf0-cc1.0-v10000_2",
            "194837_SP-uNone-d1-p1-w1-c1-f0-bf1-cc1.0-v10000_2",
            "100846_SP-uNone-d1-p1-w1-c1-f0-bf0-cc0.9999-v10000_2",
            "101142_SP-uNone-d1-p1-w1-c1-f0-bf1-cc0.9999-v10000_2",
        ]
        data_eval = [
           "books_sv_epub_100.jsonl",
        ]
        vocab_sizes = None
    if 0:
        name = "bf-bc"
        models = [
            "123500_SP-uNone-d1-p1-w1-c1-f0-bf1-cc0.9999-x1-v51200_3all-a1.0",
            "132109_SP-uNone-d1-p1-w1-c1-f0-bf0-cc0.9999-x1-v51200_3all-a1.0",
            "161742_SP-uNone-d1-p1-w1-c1-f0-bf1-cc1.0-x1-v51200_3all-a1.0",
            "185909_SP-uNone-d1-p1-w1-c1-f0-bf0-cc1.0-x1-v51200_3all-a1.0",
        ]
        data_eval = [
            "wiki_da_t1p.jsonl",
            "wiki_en_t1p.jsonl",
            "wiki_is_t1p.jsonl",
            "wiki_no_t1p.jsonl",
            "wiki_sv_t1p.jsonl",
        ]
        vocab_sizes = None
    if 0:
        name = "all-a1.0"
        models = [
            "123500_SP-uNone-d1-p1-w1-c1-f0-bf1-cc0.9999-x1-v51200_3all-a1.0",
            "180630_SP-uNone-d1-p1-w1-c1-f0-bf1-cc0.9999-x1-v64000_3all-a1.0",
            "191536_SP-uNone-d1-p1-w1-c1-f0-bf1-cc0.9999-x1-v80000_3all-a1.0",
            "204641_SP-uNone-d1-p1-w1-c1-f0-bf1-cc0.9999-x1-v96000_3all-a1.0",
            "061243_SP-uNone-d1-p1-w1-c1-f0-bf1-cc0.9999-x1-v112000_3all-a1.0",
            "080856_SP-uNone-d1-p1-w1-c1-f0-bf1-cc0.9999-x1-v128000_3all-a1.0",
        ]
        data_eval = [
            "wiki_da_t1p.jsonl",
            "wiki_en_t1p.jsonl",
            "wiki_is_t1p.jsonl",
            "wiki_no_t1p.jsonl",
            "wiki_sv_t1p.jsonl",
        ]
        vocab_sizes = None

    if 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("--name", type=str, required=True)
        parser.add_argument("--vocab_sizes", nargs='+', type=int, default=[])
        _args = parser.parse_args()

        name = _args.name
        models = get_models(name)
        vocab_sizes = _args.vocab_sizes

    main(name, models, vocab_sizes)


