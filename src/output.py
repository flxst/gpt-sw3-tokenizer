import json
from collections import Counter
from os import makedirs
from typing import List
from os.path import getsize, join
from datasets import Dataset
import sentencepiece as spm


class Output:

    def __init__(self,
                 path: str,
                 library: str):
        """
        Args:
            path: e.g. 'output/125842_[..]
            library: e.g. 'SP'
        """
        self.path = path
        self.library = library
        makedirs(self.path, exist_ok=False)

        self.parameters_file = join(self.path, "parameters.txt")
        self.tokenizer_file = join(self.path, "tokenizer.json")
        self.vocab_file = join(self.path, "tokenizer_vocab.json")
        self.merge_file = join(self.path, "tokenizer_merge.txt")
        self.subword_lengths_file = join(self.path, "tokenizer_subword_lengths.json")

        # SP
        self.model_prefix = join(self.path, "model")
        self.model_file = join(self.path, "model.model")

    def export_parameters(self, parameters) -> None:
        """ export parameters to file

        e.g. 'output/125842/
              => parameters file = 'output/125842/parameters.txt'
        """
        with open(self.parameters_file, "w") as f:
            for k, v in parameters.__dict__.items():
                if not k.startswith("__") and not callable(v):
                    f.write(f"{k} = {v}\n")

    def export_tokenizer_for_megatron_lm(self) -> None:
        """ export tokenizer vocabulary and merge rules for use with Megatron-LM

            e.g. path = 'output/125842/
                 => export files = 'output/125842/tokenizer_merge.txt'
                                   'output/125842/tokenizer_vocab.json'
        """

        if self.library == "HF":
            # a. load tokenizer_file json
            with open(self.tokenizer_file, "r", encoding="utf-8") as file:
                r = json.load(file)
            vocab = r["model"]["vocab"]
            merges = r["model"]["merges"]
        elif self.library == "SP":
            sp = spm.SentencePieceProcessor()
            sp.Load(self.model_file)
            vocab = {sp.IdToPiece(_id): _id for _id in range(sp.GetPieceSize())}
            merges = []  # use script_merge.py to extract merges from vocab
        else:
            raise Exception(f"library = {self.library} unknown, should be HF or SP")

        # b. export vocab
        with open(self.vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(vocab))
        print(f"> wrote vocab  file '{self.vocab_file}': #vocab = {len(vocab)}")

        # c. export merge
        if len(merges) > 0:
            with open(self.merge_file, "w", encoding="utf-8") as f:
                for i in range(len(merges)):
                    f.write(merges[i] + "\n")
            print(f"> wrote merges file '{self.merge_file}': #merges = {len(merges)}")

    def analyze_vocabulary(self) -> None:
        """ analyze vocabulary w.r.t. vocabulary size & subword token length

            e.g. 'output/125842/
                  => vocab file = 'output/125842/tokenizer_vocab.json'
        """

        # a. get subword lengths = dict w/ keys = subword length, value = occurrences in vocabulary
        with open(self.vocab_file, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        subwords = list(vocab.keys())
        subword_lengths_list = [len(elem) for elem in subwords]
        subword_lengths = dict(Counter(subword_lengths_list))
        assert sum(subword_lengths.values()) == len(vocab), f"ERROR! {sum(subword_lengths.values())} != {len(vocab)}"
        subword_lengths["mean"] = sum([k*v for k, v in subword_lengths.items()])/float(len(vocab))
        subword_lengths["vocab_size"] = len(vocab)

        # b. export subword lengths
        with open(self.subword_lengths_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(subword_lengths))
        print(f"> wrote subword length file '{self.subword_lengths_file}'")

    def overview(self,
                 _dataset_combined: Dataset,
                 _dataset_files: List[str],
                 _time: str) -> None:
        """ overview:
            - analyze datasets
            - time

        Used Attr:
            path: e.g. 'output/125842/ =>
                        datasets file = 'output/125842/datasets.json'

        Args:
            _dataset_combined:
            _dataset_files:
            _time:
        """
        print("-------------------")
        # a. get document count
        _overview = {
            "files": len(_dataset_combined),
            "documents_total": len(_dataset_combined["train"]),
            "documents": len(_dataset_combined["train"]),
            "dataset_files": _dataset_files,
            "data_size_total": f"{sum([getsize(_dataset_file) for _dataset_file in _dataset_files])/1073741824.:.4f}G",
            "data_size": [f"{getsize(_dataset_file)/1073741824.:.4f}G" for _dataset_file in _dataset_files],
            "time": _time,
        }

        # b. export overview
        _overview_file = join(self.path, "overview.json")
        with open(_overview_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(_overview))
        print(f"> wrote overview file '{_overview_file}'")
