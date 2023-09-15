import os
import csv
from typing import List
from os.path import join, isfile
import time
from src.env import Env
from src.helpers import LIST_OF_SPECIAL_TOKENS

env = Env()


class Parameters:
    """
    class that contains all parameters for tokenizer training
    """
    def __init__(self,
                 library: str,
                 tokenizer_name: str,
                 dataset_files: List[str],
                 dataset_filter: str = "all",
                 unicode_normalization: str = "NFC",
                 individual_digits: bool = True,
                 add_prefix_space: bool = True,
                 add_whitespace_tokens: int = 2,
                 add_code_tokens: int = 1,
                 add_newline_token: int = 0,
                 minimum_frequency: int = 0,
                 initial_alphabet: int = 1,
                 byte_fallback: bool = True,
                 character_coverage: float = 1.0,
                 train_extremely_large_corpus: bool = True,
                 vocab_size: int = 100,
                 alpha: float = 1.0):

        assert library in ["HF", "SP"], f"ERROR! library = {library} unknown, needs to be HF or SP"
        assert len(tokenizer_name), \
            "ERROR! need to specify --tokenizer_name <str>"
        assert len(dataset_files), \
            "ERROR need to specify --dataset_files <str>"
        self.library = library
        if dataset_files == ["all"]:
            self.dataset_files = get_dataset_files_in_folder(env.data_train, dataset_filter)
        else:
            self.dataset_files = [join(env.data_train, dataset_file) for dataset_file in dataset_files]
        self.tokenizer_name = tokenizer_name
        self.unicode_normalization = unicode_normalization
        self.individual_digits = bool(individual_digits)
        self.add_prefix_space = bool(add_prefix_space)
        self.add_whitespace_tokens = add_whitespace_tokens
        self.add_code_tokens = add_code_tokens
        self.add_newline_token = add_newline_token
        self.minimum_frequency = minimum_frequency
        self.initial_alphabet = initial_alphabet
        self.byte_fallback = bool(byte_fallback) if self.library == "SP" else 0
        self.character_coverage = character_coverage if self.library == "SP" else 0
        self.train_extremely_large_corpus = bool(train_extremely_large_corpus) if self.library == "SP" else 0
        self.vocab_size = vocab_size
        self.vocab_size_external = vocab_size
        self.alpha = alpha

        # DERIVED
        self.special_tokens: List[str] = []  # ["<|endoftext|>"]

        if self.add_whitespace_tokens == 1:
            whitespace_token = " " if self.library == "HF" else "▁"
            whitespace_tokens = [
                whitespace_token * i
                for i in range(2, 25)  # 2-24 consecutive whitespaces
            ]
            self.special_tokens += whitespace_tokens
        elif self.add_whitespace_tokens == 2:  # only self.library == "SP"
            assert self.library == "SP", f"ERROR! --add_whitespace_tokens 2 is only available for --library SP"
            self.vocab_size -= len(LIST_OF_SPECIAL_TOKENS)

        if self.add_code_tokens == 1:
            self.special_tokens += get_code_tokens()

        if self.add_newline_token:
            self.special_tokens += "\n"

        self.timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())[2:]

        suffix = f"_{self.tokenizer_name}-a{self.alpha}" if self.alpha != -1 else f"_{self.tokenizer_name}"
        self.output_dir = join(env.output, self.timestamp) + self.get_id() + suffix

    def show(self) -> None:
        """ print parameters """
        print("=== PARAMETERS ===")
        print(f"> library = {self.library}")
        print(f"> unicode_normalization = {self.unicode_normalization}")
        print(f"> individual_digits = {self.individual_digits}")
        print(f"> add_prefix_space = {self.add_prefix_space}")
        print(f"> add_whitespace_tokens = {self.add_whitespace_tokens}")
        print(f"> add_code_tokens = {self.add_code_tokens}")
        print(f"> add_newline_token = {self.add_newline_token}")
        print(f"> minimum_frequency = {self.minimum_frequency}")
        print(f"> initial_alphabet = {self.initial_alphabet}")
        print(f"> byte_fallback = {self.byte_fallback}")
        print(f"> character_coverage = {self.character_coverage}")
        print(f"> train_extremely_large_corpus = {self.train_extremely_large_corpus}")
        print(f"> vocab_size = {self.vocab_size_external}")
        print(f"> alpha = {self.alpha}")
        print("==================")
        print(f"> special_tokens = {self.special_tokens}")
        print("==================")
        print()

    def get_id(self) -> str:
        """
        Returns:
            id string, e.g. '-v64000'
        """
        return f"-v{self.vocab_size_external}"


def get_dataset_files_in_folder(_folder: str,
                                _dataset_filter: str) -> List[str]:
    """
    get all dataset files in folder _folder that contain _dataset_filter as substring

    Args:
        _folder: e.g. 'data_train'
        _dataset_filter: 'file'

    Returns:
        files_in_folder: e.g. ['file1.jsonl', 'file2.jsonl']
    """
    print(f"> get files in {_folder}")
    _files = [elem for elem in os.listdir(_folder) if isfile(join(_folder, elem)) and elem.endswith(".jsonl")]
    if len(_dataset_filter) and _dataset_filter != "all":
        _files = [elem for elem in _files if _dataset_filter in elem]
    assert len(_files), f"ERROR! no files found in {_folder} (that contain '{_dataset_filter}')"
    return [join(_folder, elem) for elem in _files]


def get_code_tokens() -> List[str]:
    with open('CODE_TOKENS.csv', newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        code_tokens = [row for row in csvreader][0]
    return code_tokens
