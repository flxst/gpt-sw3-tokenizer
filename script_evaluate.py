
from os.path import isfile, join, isdir
import json
# from transformers import PreTrainedTokenizerFast
import sentencepiece as spm
from collections import Counter
import time
from itertools import product
from typing import Tuple

DATA_DIR = "../data"
OUTPUT_DIR = "../output"
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

        _unk_rate = number_of_unk/float(sentence_length_in_subwords)
        _closeness_to_character_level = float(sentence_length_in_subwords)/sentence_length_in_characters

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


if __name__ == "__main__":
    if 0:
        NAME = "bf-cc-test"
        MODELS = [
            "output/194127_SP-uNone-d1-p1-w1-c1-f0-bf0-cc1.0-v10000_2",
            "output/194837_SP-uNone-d1-p1-w1-c1-f0-bf1-cc1.0-v10000_2",
            "output/100846_SP-uNone-d1-p1-w1-c1-f0-bf0-cc0.9999-v10000_2",
            "output/101142_SP-uNone-d1-p1-w1-c1-f0-bf1-cc0.9999-v10000_2",
        ]
        DATA = [
           "data/books_sv_epub_100.jsonl",
        ]
    if 1:
        NAME = "bf-bc"
        MODELS = [
            "output/123500_SP-uNone-d1-p1-w1-c1-f0-bf1-cc0.9999-x1-v51200_3all-a1.0",
        ]
        DATA = [
            "data/wiki_da_t1p.jsonl",
            "data/wiki_en_t1p.jsonl",
            "data/wiki_is_t1p.jsonl",
            "data/wiki_no_t1p.jsonl",
            "data/wiki_sv_t1p.jsonl",
        ]
    if 0:
        NAME = "all-a1.0"
        MODELS = [
            "output/123500_SP-uNone-d1-p1-w1-c1-f0-bf1-cc0.9999-x1-v51200_3all-a1.0",
            "output/180630_SP-uNone-d1-p1-w1-c1-f0-bf1-cc0.9999-x1-v64000_3all-a1.0",
            "output/191536_SP-uNone-d1-p1-w1-c1-f0-bf1-cc0.9999-x1-v80000_3all-a1.0",
            "output/204641_SP-uNone-d1-p1-w1-c1-f0-bf1-cc0.9999-x1-v96000_3all-a1.0",
            "output/061243_SP-uNone-d1-p1-w1-c1-f0-bf1-cc0.9999-x1-v112000_3all-a1.0",
            "output/080856_SP-uNone-d1-p1-w1-c1-f0-bf1-cc0.9999-x1-v128000_3all-a1.0",
        ]
        DATA = [
            "data/wiki_da_t1p.jsonl",
            "data/wiki_en_t1p.jsonl",
            "data/wiki_is_t1p.jsonl",
            "data/wiki_no_t1p.jsonl",
            "data/wiki_sv_t1p.jsonl",
        ]

    results = {
        model: {
            data: dict()
            for data in DATA
        }
        for model in MODELS
    }

    for model, data in product(MODELS, DATA):
        # bf, cc = extract_bf_cc_from_model(model)
        # print()
        # print(f"> model = {model}")
        # print(f"> data = {data}")
        # print(".", end="")
        Xunk_rate, Xcloseness_to_character_level = evaluate(model, data)
        results[model][data]["unk_rate"] = Xunk_rate
        results[model][data]["closeness_to_character_level"] = Xcloseness_to_character_level

    print()
    print("---------------")
    print(results)

    results_file = f"output/evaluation/results_{NAME}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(results))