from os.path import isfile, join, isdir
from transformers import PreTrainedTokenizerFast
import sentencepiece as spm
from collections import Counter
import time
import string
import json
from typing import Dict

from src.env import Env
from src.evaluation.evaluation_metrics import EvaluationMetrics

env = Env()
REMOVE_PUNCTUATION = True


def evaluate(_tokenizer: str, _data_path: str) -> EvaluationMetrics:
    """
    evaluate single tokenizer on single evaluation dataset

    Args:
        _tokenizer: e.g. <output>/151508_SP-uNone-d0-p0-w0-c0-f0-bf0-cc1.0-x1-v1000_SP_test
        _data_path: e.g. <data_eval>/all_en.jsonl'

    Returns:
        evaluation_metrics: [EvaluationMetrics]
    """
    assert isdir(_tokenizer), f"ERROR! tokenizer = {_tokenizer} does not exist."
    ts = time.time()

    # 0. load data
    with open(_data_path, "r", encoding="utf-8") as file:
        _data = [json.loads(line)["text"] for line in file]

    evaluation_metrics = EvaluationMetrics()

    # 1. tokenize data
    if isfile(join(_tokenizer, "tokenizer.json")):
        library = "HF"
        tokenizer_file = join(_tokenizer, "tokenizer.json")
        tokenizer_fast = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
    elif isfile(join(_tokenizer, "model.model")):
        library = "SP"
        tokenizer_file = join(_tokenizer, "model.model")
        sp = spm.SentencePieceProcessor(model_file=tokenizer_file)
    else:
        raise Exception(f"ERROR! model seems to be neither in SP or HF format.")

    number_of_unk = 0
    sentence_length_in_subwords = 0
    sentence_length_in_characters = 0
    subwords_b = 0  # beginning of word, i.e. starting with "_" (SP) or "Ġ" (HF)
    subwords_i = 0  # interior of word, i.e. not starting with "_" (SP) or "Ġ" (HF)
    subwords_b_proportion = 0
    subwords_i_proportion = 0
    token_frequencies: Dict[int, int] = dict()

    for i, example in enumerate(_data):
        if REMOVE_PUNCTUATION:
            example = example.translate(str.maketrans("", "", string.punctuation))

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
        number_of_unk += counter[1]  # 1 = <unk>  TODO

        # ctcl
        sentence_length_in_characters += len(example)

        # fertility
        new_word_character = "▁" if library == "SP" else "Ġ"
        encoding_mask = [
            1 if elem.startswith(new_word_character) else 0 for elem in encoding_str
        ]
        subwords_b += sum(encoding_mask)
        subwords_i += len(encoding_mask) - sum(encoding_mask)

        # proportion
        encoding_mask_proportion = list()
        for j in range(len(encoding_mask)):
            current_elem = encoding_mask[j]
            previous_elem = encoding_mask[j - 1]
            if j == 0:
                encoding_mask_proportion.append(current_elem)
            elif current_elem == 1:
                encoding_mask_proportion.append(current_elem)
            elif current_elem == 0 and previous_elem == 1:
                encoding_mask_proportion.append(current_elem)

        subwords_b_proportion += sum(encoding_mask_proportion)
        subwords_i_proportion += len(encoding_mask_proportion) - sum(
            encoding_mask_proportion
        )

        # token frequencies
        for k, v in dict(counter).items():
            if k not in token_frequencies.keys():
                token_frequencies[k] = v
            else:
                token_frequencies[k] += v

        # debug
        if env.debug:
            print(example[:50])
            print(encoding[:10])
            print(counter[0])
            # print(example_encoded[:10])
            print("STOP (env.debug = True).")
            break

        assert subwords_b + subwords_i == sentence_length_in_subwords, (
            f"ERROR! subwords_b + subwords_i = {subwords_b} + {subwords_i} "
            f"is not equal to sentence_length_in_subwords = {sentence_length_in_subwords}"
        )
        assert (
            subwords_b == subwords_b_proportion
        ), f"ERROR! subwords_b = {subwords_b} is not equal to subwords_b_proportion = {subwords_b_proportion}"

        evaluation_metrics.set(
            "unk_rate",
            {"nominator": number_of_unk, "denominator": sentence_length_in_subwords},
        )
        evaluation_metrics.set(
            "ctcl",
            {
                "nominator": sentence_length_in_subwords,
                "denominator": sentence_length_in_characters,
            },
        )
        evaluation_metrics.set(
            "fertility",
            {"nominator": subwords_b + subwords_i, "denominator": subwords_b},
        )
        evaluation_metrics.set(
            "proportion",
            {"nominator": subwords_i_proportion, "denominator": subwords_b_proportion},
        )
        evaluation_metrics.set(
            "token_frequencies", {"value": dict(sorted(token_frequencies.items()))}
        )

        if env.verbose:
            print()
            print(f"[time = {time.time() - ts:.2f}s]")
            print()
            print(f"> number_of_unk = {number_of_unk}")
            print(f"> sentence_length_in_subwords = {sentence_length_in_subwords}")
            print(f"> sentence_length_in_characters = {sentence_length_in_characters}")
            print()
            print(f"=> unk rate = {evaluation_metrics.unk_rate:.3f}")
            print(f"=> closeness to character level = {evaluation_metrics.ctcl:.3f}")
            print(f"=> fertility = {evaluation_metrics.fertility:.3f}")

    return evaluation_metrics
