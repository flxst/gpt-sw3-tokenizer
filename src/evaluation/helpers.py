
from typing import Dict, List
import os
import json
from os.path import join, isdir
from src.env import Env

env = Env()


def get_tokenizer(_tokenizer_name: str) -> str:
    """
    get full tokenizer path that corresponds to _tokenizer_name

    Args:
        _tokenizer_name: e.g. 'SP_test'

    Returns:
        tokenizer: e.g. <output>/151508_SP-uNone-d0-p0-w0-c0-f0-bf0-cc1.0-x1-v1000_SP_test
    """
    subdirs = [
        elem
        for elem in os.listdir(env.output)
        if isdir(join(env.output, elem)) and elem.endswith(_tokenizer_name)
    ]
    assert len(subdirs) > 0, f"ERROR! did not find any subdirectories that end " \
                             f"with {_tokenizer_name} in env.output = {env.output}"
    assert len(subdirs) == 1, f"ERROR! found multiple subdirectories: {subdirs}"
    _tokenizer_name_complete = join(env.output, subdirs[0])
    return _tokenizer_name_complete


def get_vocab_size(_tokenizer_name_complete: str) -> str:
    try:
        return _tokenizer_name_complete.split("-v")[-1].split("_")[0]
    except Exception:
        raise Exception(f"ERROR! could not retrive vocabulary size from tokenizer name = {_tokenizer_name_complete}")


def instantiate_nested_dict(list1: List[str], list2: List[str]) -> Dict[str, Dict[str, Dict]]:
    return {
        elem1: {
            elem2: dict()
            for elem2 in list2
        }
        for elem1 in list1
    }


def write_json(_dict: Dict, file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(_dict))
    print(f"> wrote json file {file_path}")

