"""
EXECUTION: python script_create_merge_file_sp.py
           --tokenizer_directory <tokenizer_directory>

PURPOSE: the script
         - loads the tokenizer from <output>/<tokenizer_directory>/model.model
         - writes a merge file to <output>/<tokenizer_directory>/tokenizer_merge.txt
"""
import argparse
from os.path import join

from os.path import abspath, dirname
import sys
BASE_DIR = abspath(dirname(dirname(dirname(abspath(__file__)))))
print(f">>> BASE_DIR: {BASE_DIR}")
sys.path.append(BASE_DIR)

# from src.helpers import create_merge_rules
from src.env import Env


import json
import sentencepiece as spm
from typing import Tuple, Dict, List, Optional
from tqdm import trange, tqdm


class SentencePieceExtractor:
    """
    see https://github.com/huggingface/tokenizers/blob/main/bindings/python/scripts/sentencepiece_extractor.py

    Extractor implementation for SentencePiece trained models.
    https://github.com/google/sentencepiece
    """

    def __init__(self, model: Optional[str] = None):
        if model is not None:
            # Get SentencePiece
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(model)
        else:
            self.sp = None

    def get_vocab(self):
        assert self.sp is not None, f"ERROR! self.sp is None!"
        sp = self.sp
        _vocab = {sp.id_to_piece(index): index for index in trange(sp.GetPieceSize())}
        return _vocab

    def extract(self, vocab) -> Tuple[Dict[str, int], List[Tuple]]:
        # Merges
        merges = []
        for piece_l in tqdm(vocab.keys(), total=len(vocab)):  # sp.GetPieceSize()):
            for piece_r in vocab.keys():
                merge = f"{piece_l}{piece_r}"
                piece_id = vocab.get(merge, None)
                if piece_id:
                    merges += [(piece_l, piece_r, piece_id)]
        merges = sorted(merges, key=lambda val: val[2])
        merges = [(val[0], val[1]) for val in merges]

        return merges


def main(_args):
    env = Env()
    # model_name = "180051_SP-uNone-d1-p1-w2-c1-f0-bf1-cc0.9999-x1-v64000_tokenizer2"
    model_name = _args.tokenizer_directory

    ############
    # approach 1
    ############
    vocab_file = join(env.output, model_name, "tokenizer_vocab.json")
    merge_file = join(env.output, model_name, "tokenizer_merge.txt")

    # _ = create_merge_rules(vocab_file, merge_file)

    ############
    # approach 2
    ############
    if "_SP" in model_name:
        model_file = join(env.output, model_name, "model.model")
        spe = SentencePieceExtractor(model_file)
        vocab = spe.get_vocab()
        merges = spe.extract(vocab)
    elif "_HF" in model_name:
        with open(vocab_file, "r") as f:
            vocab = json.load(f)
        print(f"> read vocab with {len(vocab)} words from {vocab_file}")
        spe = SentencePieceExtractor()
        merges = spe.extract(vocab)
    else:
        raise Exception(f"ERROR! could not find _HF or _SP in model_name = {model_name}")

    # write merge_file
    with open(merge_file, "w") as f:
        for merge in merges:
            f.write(f"{merge[0]} {merge[1]}\n")
    print(f"> wrote {len(merges)} merges to {merge_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_directory", type=str, required=True)
    _args = parser.parse_args()

    main(_args)

