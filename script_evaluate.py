"""
EXECUTION: python script_evaluate.py
           --tokenizer_name <tokenizer_name>          # e.g. example
           --vocab_size <vocab_size>                  # e.g. 64000
           [--vocab_size_pruned <vocab_size_pruned>]  # e.g. 40000 51200
           [--monolingual]                            # if used, monolingual models are evaluated

PURPOSE: the script
     - creates tokenizers with pruned vocabularies if multiple vocab_sizes are specified
       (note that the tokenizer with the last specified vocab size has to exist)
     - applies each tokenizer on each dataset and computes evaluation metrics
       (unk_rate, ctcl, fertility, proportion of continued words, token_frequencies)
     - writes results to
        - `<output>/evaluation/results_*.json`
        - `<output>/evaluation/token_frequencies_*.json`
"""

import os
from os.path import join
import argparse
from itertools import product
from typing import List, Optional
from src.env import Env
from src.helpers import get_languages
from src.analysis import _analyze_vocab, extract_vocab
from src.evaluation.helpers import get_tokenizer, instantiate_nested_dict, write_json
from src.evaluation.evaluate import evaluate
from src.evaluation.prune_vocab_size import prune_vocab_size

env = Env()


def main(_tokenizer_name: str, _vocab_size: int, _vocab_size_pruned: Optional[List[int]] = None):
    _tokenizer = get_tokenizer(_tokenizer_name)
    _data_eval = [join(env.data_eval, elem) for elem in os.listdir(env.data_eval)]

    # _vocab_sizes = _vocab_size_pruned + [_vocab_size]
    if _vocab_size_pruned is None:
        _vocab_sizes = []
    else:
        for elem in _vocab_size_pruned:
            assert elem < _vocab_size, f"ERROR! pruned vocab size {elem} needs to be < than vocab size {_vocab_size}"
        _vocab_sizes = _vocab_size_pruned

    _vocab_sizes += [_vocab_size]

    # prune vocabulary -> tokenizers
    last_regular_token_index = _analyze_vocab(env, _tokenizer)["merges"][-1]  # get index of last regular token
    tokenizers = prune_vocab_size(_tokenizer, _vocab_sizes, last_regular_token_index)  # returns list

    # extract vocabulary files for pruned tokenizers
    print(f"> extract vocab for {len(tokenizers)-1} pruned tokenizers")
    for tokenizer in tokenizers[1:]:
        tokenizer_path = join(tokenizer, "model.model")
        extract_vocab(tokenizer_path)  # TODO: only implemented (needed?) for SP

    results = instantiate_nested_dict(tokenizers, _data_eval)
    token_frequencies = instantiate_nested_dict(tokenizers, _data_eval)

    for tokenizer, data in product(tokenizers, _data_eval):
        if env.debug:
            print()
            print(f"> tokenizer = {tokenizer}")
            print(f"> data = {data}")
        evaluation_metrics = evaluate(tokenizer, data)
        results[tokenizer][data] = evaluation_metrics.as_dict()
        token_frequencies[tokenizer][data] = results[tokenizer][data].pop("token_frequencies")

    print("\n--- results ---")
    print(results)
    print("---------------")

    os.makedirs(join(env.output, "evaluation"), exist_ok=True)

    results_file = join(env.output, "evaluation", f"results_{_tokenizer_name}.json")
    write_json(results, results_file)

    token_frequencies_file = join(env.output, "evaluation", f"token_frequencies_{_tokenizer_name}.json")
    write_json(token_frequencies, token_frequencies_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--vocab_size_pruned", nargs='+', type=int, default=[])
    parser.add_argument("--monolingual", action="store_true")
    _args = parser.parse_args()

    tokenizer_name = _args.tokenizer_name
    vocab_size = _args.vocab_size
    vocab_size_pruned = _args.vocab_size_pruned

    monolingual = _args.__dict__.pop("monolingual")
    if monolingual is False:
        main(tokenizer_name, vocab_size, vocab_size_pruned)
    else:
        for language in get_languages("eval"):
            tokenizer_name += f"_{language}"
            main(tokenizer_name, vocab_size, vocab_size_pruned)
