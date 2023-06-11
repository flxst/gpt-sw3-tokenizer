"""
EXECUTION: python script_evaluate.py --tokenizer_name example --vocab_sizes 51200 64000

PURPOSE: the script
         - applies each tokenizer on each dataset and computes metrics
           (unk_rate, ctcl, fertility, proportion of continued words, token_frequencies)
         - writes results to `<OUTPUT>/evaluation/results_*.json`
"""

from copy import deepcopy
import os
from os.path import isfile, join, isdir
import json
import argparse
from transformers import PreTrainedTokenizerFast
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
DATA_DIR = env.data_train
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
        "token_frequencies": None,
    }

    # 1. tokenize data
    if isfile(join(_model_dir, "tokenizer.json")):
        library = "HF"
        tokenizer_file = join(_model_dir, "tokenizer.json")
        tokenizer_fast = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
    elif isfile(join(_model_dir, "model.model")):
        library = "SP"
        tokenizer_file = join(_model_dir, "model.model")
        sp = spm.SentencePieceProcessor(model_file=tokenizer_file)
    else:
        raise Exception(f"ERROR! model seems to be neither in SP or HF format.")

    number_of_unk = 0
    sentence_length_in_subwords = 0
    sentence_length_in_characters = 0
    subwords_b = 0  # beginning of word, i.e. starting with _
    subwords_i = 0  # interior of word, i.e. not starting with _
    subwords_b_proportion = 0
    subwords_i_proportion = 0
    token_frequencies = dict()

    for i, example in enumerate(_data):

        if REMOVE_PUNCTUATION:
            example = example.translate(str.maketrans('', '', string.punctuation))

        # general
        if library == "SP":
            encoding = sp.encode(example, out_type=int)
            encoding_str = sp.encode(example, out_type=str)
        else:  # HF
            encoding = tokenizer_fast.encode(example)
            encoding_str = tokenizer_fast.tokenize(example)

        sentence_length_in_subwords += len(encoding)  # for unk, ctcl

        # unk
        counter = Counter(encoding)
        number_of_unk += counter[1]  # 1 = <unk>

        # ctcl
        sentence_length_in_characters += len(example)

        # fertility
        new_word_character = "▁" if library == "SP" else "Ġ"
        encoding_mask = [1 if elem.startswith(new_word_character) else 0 for elem in encoding_str]
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

        # token frequencies
        for k, v in dict(counter).items():
            if k not in token_frequencies.keys():
                token_frequencies[k] = v
            else:
                token_frequencies[k] += v

        # debug
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

        try:
            _metrics["token_frequencies"] = dict(sorted(token_frequencies.items()))
        except ZeroDivisionError:
            _metrics["token_frequencies"] = -1

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
    """only works for library == SP"""  # TODO
    _new_models = [_models]

    if len(_vocab_sizes) > 1:
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
    print(f"> extract vocab for {len(_models)-1} pruned models")
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
    token_frequencies = {
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
        token_frequencies[model][data] = results[model][data].pop("token_frequencies")

    print("\n--- results ---")
    print(results)
    print("---------------")

    os.makedirs(join(env.output, "evaluation"), exist_ok=True)
    results_file = join(env.output, "evaluation", f"results_{_tokenizer_name}.json")
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(results))
    print(f"> wrote results to {results_file}")

    for model, data in product(_models, _data_eval):
        # model_name = model.split("/")[-1].strip(".json")
        data_name = data.split("/")[-1].split(".jsonl")[0]
        tokens_distribution_file = join(env.output, "evaluation", f"token_frequencies_{_tokenizer_name}_{data_name}.json")
        with open(tokens_distribution_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(token_frequencies[model][data]))
        print(f"> wrote results to {tokens_distribution_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", type=str, required=True)
    parser.add_argument("--vocab_sizes", nargs='+', type=int, default=[])
    _args = parser.parse_args()

    tokenizer_name = _args.tokenizer_name
    models = get_models(tokenizer_name)
    vocab_sizes = _args.vocab_sizes

    main(tokenizer_name, models, vocab_sizes)
