import json
from tokenizers import normalizers
from datasets import Dataset
from typing import List
from os.path import getsize

UNICODE_NORMALIZATION = {
    "NFC": normalizers.NFC(),
    "NFKC": normalizers.NFKC(),
    "NFKD": normalizers.NFKD(),
}


def get_normalizer(_unicode_normalization: str):
    return UNICODE_NORMALIZATION[_unicode_normalization]


def export_tokenizer_for_megatron_lm(_tokenizer_file: str) -> None:
    """ export tokenizer vocabulary and merge rules for use with Megatron-LM

    Args:
        _tokenizer_file: e.g. 'output/125842/tokenizer => export files =
                              'output/125842/tokenizer_merge.txt'
                              'output/125842/tokenizer_vocab.json'
    """

    # a. load tokenizer_file json
    with open(_tokenizer_file + ".json", "r") as file:
        r = json.load(file)
    vocab = r["model"]["vocab"]
    merges = r["model"]["merges"]

    # b. export vocab
    _vocab_file = f"{_tokenizer_file}_vocab.json"
    with open(_vocab_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(vocab))
    print(f"> wrote vocab  file '{_vocab_file}': #vocab = {len(vocab)}")

    # c. export merge
    _merge_file = f"{_tokenizer_file}_merge.txt"
    with open(_merge_file, "w", encoding="utf-8") as f:
        for i in range(len(merges)):
            f.write(merges[i] + "\n")
    print(f"> wrote merges file '{_merge_file}': #merges = {len(merges)}")


def analyze_vocabulary(_tokenizer_file: str) -> None:
    """ analyze vocabulary w.r.t. vocabulary size & subword token length

    Args:
        _tokenizer_file: e.g. 'output/125842/tokenizer =>
                              vocab file = 'output/125842/tokenizer_vocab.json'
    """
    from collections import Counter

    # a. get subword lengths = dict w/ keys = subword length, value = occurrences in vocabulary
    _vocab_file = f"{_tokenizer_file}_vocab.json"
    with open(_vocab_file, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    subwords = list(vocab.keys())
    subword_lengths_list = [len(elem) for elem in subwords]
    subword_lengths = dict(Counter(subword_lengths_list))
    assert sum(subword_lengths.values()) == len(vocab), f"ERROR! {sum(subword_lengths.values())} != {len(vocab)}"
    subword_lengths["mean"] = sum([k*v for k, v in subword_lengths.items()])/float(len(vocab))
    subword_lengths["vocab_size"] = len(vocab)

    # b. export subword lengths
    _subword_lengths_file = f"{_tokenizer_file}_subword_lengths.json"
    with open(_subword_lengths_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(subword_lengths))
    print(f"> wrote subword length file '{_subword_lengths_file}'")


def overview(_tokenizer_file: str,
             _datasets: List[Dataset],
             _data_files: List[str],
             _time: str) -> None:
    """ overview:
        - analyze datasets
        - time

    Args:
        _tokenizer_file: e.g. 'output/125842/tokenizer =>
                              datasets file = 'output/125842/datasets.json'
        _datasets:
        _data_files:
        _time:
    """
    print("-------------------")
    # a. get document count
    _overview = {
        "files": len(_datasets),
        "documents_total": sum([len(dataset["train"]) for dataset in _datasets]),
        "documents": [len(dataset["train"]) for dataset in _datasets],
        "data_files": _data_files,
        "data_size_total": f"{sum([getsize(_data_file) for _data_file in _data_files])/1073741824.:.4f}G",
        "data_size": [f"{getsize(_data_file)/1073741824.:.4f}G" for _data_file in _data_files],
        "time": _time,
    }

    # b. export overview
    _overview_file = f"{_tokenizer_file[:-10]}/overview.json"
    with open(_overview_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(_overview))
    print(f"> wrote overview file '{_overview_file}'")


