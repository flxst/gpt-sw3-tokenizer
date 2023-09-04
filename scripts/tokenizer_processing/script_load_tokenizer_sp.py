"""
EXECUTION: python script_load_tokenizer_sp.py
           --tokenizer_directory 152808-v64000_tokenizer1
           [--verbose]

PURPOSE: the script
         - loads the tokenizer from <output>/<tokenizer_directory>/model.model
         - prints some infos
"""

import argparse
from os.path import join, abspath, dirname
import sys
from sentencepiece import sentencepiece_model_pb2 as model
import sentencepiece as spm
BASE_DIR = abspath(dirname(dirname(dirname(abspath(__file__)))))
print(f">>> BASE_DIR: {BASE_DIR}")
sys.path.append(BASE_DIR)

from src.env import Env


def main(_args):
    env = Env()

    # checkpoint = 'output/194046_SP-uNFC-d1-p1-w1-c1-f0-v500_2/model.model'
    checkpoint = join(env.output, _args.tokenizer_directory, 'model.model')

    print("\n===")
    sp = spm.SentencePieceProcessor()
    sp.Load(checkpoint)
    vocabs = {sp.IdToPiece(_id): _id for _id in range(sp.GetPieceSize())}
    print(f"vocabulary size = {len(vocabs)}")
    print()
    for _id in range(5):
        print(f"vocabulary item {vocabs[sp.IdToPiece(_id)]}:", sp.IdToPiece(_id), )
    for _id in range(len(vocabs)-5, len(vocabs)):
        print(f"vocabulary item {vocabs[sp.IdToPiece(_id)]}:", sp.IdToPiece(_id))

    m = model.ModelProto()
    m.ParseFromString(open(checkpoint, 'rb').read())
    if _args.verbose:
        print("\n SP pieces")
        print(m.pieces)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_directory", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")
    _args = parser.parse_args()

    main(_args)
