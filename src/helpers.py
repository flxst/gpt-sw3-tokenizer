from tokenizers import normalizers
from datasets import Dataset
import os
import shutil
from os.path import join
from sentencepiece import sentencepiece_model_pb2 as model_pb2

UNICODE_NORMALIZATION = {
    "None": None,
    "NFC": normalizers.NFC(),
    "NFKC": normalizers.NFKC(),
    "NFKD": normalizers.NFKD(),
}

LIST_OF_SPECIAL_TOKENS = [
    "▁" * i for i in range(2, 25)
]


def get_normalizer(_unicode_normalization: str):
    return UNICODE_NORMALIZATION[_unicode_normalization]


def get_training_corpus_combined(_dataset: Dataset, batch_size: int = 100000):
    for i in range(0, len(_dataset['train']), batch_size):
        yield str(_dataset['train'][i: i + batch_size]["text"])


def add_special_tokens(_model_path: str, overwrite: bool = True):

    _vocab_path = join(_model_path, "model.vocab")
    if overwrite:
        _new_model_path = _model_path
        _new_vocab_path = _vocab_path
    else:
        _new_model_path = _model_path + "___MST"
        _new_vocab_path = join(_new_model_path, "model.vocab")

    # A1. read model
    print("\n--- 1. read model ---")
    m = model_pb2.ModelProto()
    m.ParseFromString(open(join(_model_path, 'model.model'), 'rb').read())
    lowest_score = m.pieces[-1].score
    print(f"> lowest_score = {lowest_score}")

    # A2. unprioritize accidental special tokens (special tokens that happen to exist as learned pieces)
    print("\n--- 2. unprioritize accidental special tokens ---")
    counter_unprioritize = 0
    for p in m.pieces:
        if list(set(p.piece)) == ["▁"] and p.piece in LIST_OF_SPECIAL_TOKENS:
            print(f"> unprioritize {p.piece}")
            p.piece = f"UNPRIORITIZED_{counter_unprioritize}"
            p.score = lowest_score - 2
            counter_unprioritize += 1
    print(f"> unprioritized {counter_unprioritize} accidental special tokens")

    # A3. add special tokens
    print("\n--- 3. add special tokens ---")
    print(len(m.pieces))
    print(m.pieces[-1])
    for special_token in LIST_OF_SPECIAL_TOKENS:
        m.pieces.add()
        m.pieces[-1].piece = special_token
        m.pieces[-1].score = lowest_score - 1
    print(len(m.pieces))
    print(m.pieces[-1])
    print(f"> added {len(LIST_OF_SPECIAL_TOKENS)} pieces with score = {lowest_score - 1}")

    # A4. write new model
    print("\n--- 4. write new model ---")
    os.makedirs(_new_model_path, exist_ok=True)
    with open(join(_new_model_path, 'model.model'), 'wb') as f:
        f.write(m.SerializeToString())
    print(f"> wrote new model to {join(_new_model_path, 'model.model')}")

    # B1. add special tokens to vocab file
    if not overwrite:
        shutil.copyfile(_vocab_path, _new_vocab_path)

    with open(_new_vocab_path, "a") as f:
        for special_token in LIST_OF_SPECIAL_TOKENS:
            f.write(f"{special_token}\t{int(lowest_score-1)}\n")
