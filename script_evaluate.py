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
import string
from itertools import product
from typing import Tuple, List, Dict, Any
from sentencepiece import sentencepiece_model_pb2 as model_pb2
from src.env import Env
from src.analysis import _analyze_vocab, extract_vocab

env = Env()
DATA_DIR = env.data_sampled
OUTPUT_DIR = env.output
DEBUG = 0
VERBOSE = 0

REMOVE_PUNCTUATION = True


def evaluate(_model_dir, _data_path) -> Dict[str, Any]:
    """
    evaluate single model on single dataset

    Args:
        _model_dir:
        _data_path:

    Returns:
        metrics: [Dict] w/ keys = unk_rate, ctcl, fertility, proportion
    """
    assert isdir(_model_dir), f"ERROR! model_dir = {_model_dir} does not exist."
    ts = time.time()

    # 0. load data
    with open(_data_path, "r", encoding="utf-8") as file:
        _data = [json.loads(line)["text"] for line in file]

    _metrics: Dict[str, Any] = {
        "unk_rate": None,
        "ctcl": None,
        "fertility": None,
        "proportion": None,
    }

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
        subwords_b = 0  # beginning of word, i.e. starting with _
        subwords_i = 0  # interior of word, i.e. not starting with _
        subwords_b_proportion = 0
        subwords_i_proportion = 0
        for i, example in enumerate(_data):
            # if i == 0:
            #     print()
            #     print(example[:50])
            if REMOVE_PUNCTUATION:
                example = example.translate(str.maketrans('', '', string.punctuation))
            # if i == 0:
            #     print(example[:50])

            # general
            encoding = sp.encode(example, out_type=int)
            encoding_str = sp.encode(example, out_type=str)
            sentence_length_in_subwords += len(encoding)  # for unk, ctcl

            # unk
            counter = Counter(encoding)
            number_of_unk += counter[1]  # 1 = <unk>

            # ctcl
            sentence_length_in_characters += len(example)

            # fertility
            encoding_mask = [1 if elem.startswith("▁") else 0 for elem in encoding_str]
            subwords_b += sum(encoding_mask)
            subwords_i += len(encoding_mask) - sum(encoding_mask)

            # proportion
            encoding_mask_proportion = list()
            for j in range(len(encoding_mask)):
                current_elem = encoding_mask[j]
                previous_elem = encoding_mask[j-1]
                if j == 0:
                    encoding_mask_proportion.append(current_elem)
                elif current_elem == 1:
                    encoding_mask_proportion.append(current_elem)
                elif current_elem == 0 and previous_elem == 1:
                    encoding_mask_proportion.append(current_elem)

            subwords_b_proportion += sum(encoding_mask_proportion)
            subwords_i_proportion += len(encoding_mask_proportion) - sum(encoding_mask_proportion)

            # example_encoded = sp.encode(example, out_type=str)
            # assert len(example_encoded) == len(encoding)

            if DEBUG:
                print(example[:50])
                print(encoding[:10])
                print(counter[0])
                # print(example_encoded[:10])
                break

        assert subwords_b + subwords_i == sentence_length_in_subwords, \
            f"ERROR! subwords_b + subwords_i = {subwords_b} + {subwords_i} " \
            f"is not equal to sentence_length_in_subwords = {sentence_length_in_subwords}"
        assert subwords_b == subwords_b_proportion, \
            f"ERROR! subwords_b = {subwords_b} is not equal to subwords_b_proportion = {subwords_b_proportion}"

        try:
            _metrics["unk_rate"] = number_of_unk/float(sentence_length_in_subwords)
        except ZeroDivisionError:
            _metrics["unk_rate"] = -1

        try:
            _metrics["ctcl"] = float(sentence_length_in_subwords)/sentence_length_in_characters
        except ZeroDivisionError:
            _metrics["ctcl"] = -1

        try:
            _metrics["fertility"] = (float(subwords_b) + float(subwords_i)) / float(subwords_b)
        except ZeroDivisionError:
            _metrics["fertility"] = -1

        try:
            _metrics["proportion"] = float(subwords_i_proportion) / float(subwords_b_proportion)
        except ZeroDivisionError:
            _metrics["proportion"] = -1

        if VERBOSE:
            print()
            print(f"[time = {time.time() - ts:.2f}s]")
            print()
            print(f"> number_of_unk = {number_of_unk}")
            print(f"> sentence_length_in_subwords = {sentence_length_in_subwords}")
            print(f"> sentence_length_in_characters = {sentence_length_in_characters}")
            print()
            print(f"=> unk rate = {_metrics['unk_rate']:.3f}")
            print(f"=> closeness to character level = {_metrics['ctcl']:.3f}")
            print(f"=> fertility = {_metrics['fertility']:.3f}")

    return _metrics


def extract_bf_cc_from_model(_model) -> Tuple[str, str]:
    _bf = _model.split("-bf")[1].split("-cc")[0]
    _cc = _model.split("-cc")[1].split("-v")[0]
    return _bf, _cc


def get_models(_tokenizer_name: str) -> List[str]:
    subdirs = [
        elem
        for elem in os.listdir(env.output)
        if isdir(join(env.output, elem)) and elem.endswith(_tokenizer_name)
    ]
    assert len(subdirs) > 0, f"ERROR! did not find any subdirectories that end " \
                             f"with {_tokenizer_name} in env.output = {env.output}"
    assert len(subdirs) == 1, f"ERROR! found multiple subdirectories: {subdirs}"
    _models = subdirs
    return _models


def prune_vocab_size(_models: str,
                     _vocab_sizes: List[int],
                     _last_regular_token_index: int) -> List[str]:
    """only works for library == SP"""
    _new_models = [_models]

    # initial model
    vocab_size_model = _vocab_sizes[-1]
    assert str(vocab_size_model) in _models, \
        f"ERROR! vocab size = {vocab_size_model} is not in _models = {_models}"
    model_file = join(_models, "model.model")
    m = model_pb2.ModelProto()
    m.ParseFromString(open(model_file, 'rb').read())

    indices = _analyze_vocab(env, _models)
    print()
    print(f"> last regular token index: {indices['merges'][-1]}")

    # pruned models
    for _vocab_size in _vocab_sizes[:-1]:
        pruned_model_dir = _models.replace(f"v{vocab_size_model}", f"v{_vocab_size}")
        os.makedirs(pruned_model_dir, exist_ok=False)
        pruned_model = join(pruned_model_dir, "model.model")

        m_pruned = deepcopy(m)
        for i, _ in enumerate(m.pieces):
            if _last_regular_token_index - (vocab_size_model - _vocab_size) < i <= _last_regular_token_index:
                # workaround: overwrite with extremely unlikely token
                m_pruned.pieces[i].piece = f"a!?x$$▁!!xyz.masdf_{i}"

        for j in [0, 1, 2, vocab_size_model-23, vocab_size_model-22]:
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


def main(_tokenizer_name, _models, _vocab_sizes=None):
    _models = [join(env.output, model) for model in _models]
    _data_eval = [join(env.data_eval, elem) for elem in os.listdir(env.data_eval)]

    # prune vocabulary
    if len(_models) == 1:
        last_regular_token_index = _analyze_vocab(env, _models[0])["merges"][-1]  # get index of last regular token
        _models = prune_vocab_size(_models[0], _vocab_sizes, last_regular_token_index)  # returns list

    # extract vocabulary files
    print("extract vocab")
    for _model in _models[1:]:
        _model_path = join(_model, "model.model")
        extract_vocab(_model_path)

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
        results[model][data] = evaluate(model, data)

    print()
    print("--- results ---")
    print(results)
    print("---------------")

    os.makedirs(join(env.output, "evaluation"), exist_ok=True)
    results_file = join(env.output, "evaluation", f"results_{_tokenizer_name}.json")
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(results))
    print(f"> wrote results to {results_file}")


if __name__ == "__main__":
    if 0:
        tokenizer_name = "bf-cc-test"
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
        tokenizer_name = "bf-bc"
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
        tokenizer_name = "all-a1.0"
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
        parser.add_argument("--tokenizer_name", type=str, required=True)
        parser.add_argument("--vocab_sizes", nargs='+', type=int, default=[])
        _args = parser.parse_args()

        tokenizer_name = _args.tokenizer_name
        models = get_models(tokenizer_name)
        vocab_sizes = _args.vocab_sizes

    main(tokenizer_name, models, vocab_sizes)


