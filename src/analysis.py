
from os.path import join, isfile
import json
import re
from typing import Dict, List
# from sentencepiece import sentencepiece_model_pb2 as model
import sentencepiece as spm

DEBUG = 0


def _analyze_vocab(_env, _model) -> Dict[str, List[int]]:
    vocab_file = join(_model, "tokenizer_vocab.json")
    vocab_file_full = join(_env.output, vocab_file)
    assert isfile(vocab_file_full), f"ERROR! vocab_file = {vocab_file_full} does not exist."
    with open(vocab_file_full, "r") as f:
        vocab_dict = json.load(f)
    print(f"> read vocabulary of size = {len(vocab_dict)} \n  from {vocab_file}")

    vocab = list(vocab_dict.keys())

    indices = {
        key: list()
        for key in ["special", "<|*|>", "<0x*>", "merges", "pruned", "single_character", "whitespace"]
    }

    # 1. special token
    _pad = [i for i, item in enumerate(vocab) if re.search('<pad>', item)]
    _unk = [i for i, item in enumerate(vocab) if re.search('<unk>', item)]
    _s = [i for i, item in enumerate(vocab) if re.search('<s>', item)]
    _eos = [i for i, item in enumerate(vocab) if re.search('<\|endoftext\|>', item)]
    indices["special"] = _pad + _unk + _s + _eos

    # 2. <|*|> tokens
    indices["<|*|>"] = [i for i, item in enumerate(vocab) if re.search('<\|.+\|>', item) and item != '<|endoftext|>']

    # 3. <0x*> byte fallback tokens
    indices["<0x*>"] = [i for i, item in enumerate(vocab) if re.search('<0x.+>', item)]

    # X. pruned tokens
    indices["pruned"] = [i for i, item in enumerate(vocab) if re.search('a\!\?x\$\$\▁\!\!xyz\.masdf', item)]

    # 5. single character tokens
    indices["single_character"] = [i for i, item in enumerate(vocab) if len(item) == 1]

    # 6. whitespace tokens
    indices["whitespace"] = [i for i, item in enumerate(vocab) if len(item) > 1 and list(set(item)) == ['▁']]

    # 4. merges
    all_indices = list()
    for key in indices.keys():
        all_indices.extend(indices[key])
    indices["merges"] = [i for i in range(len(vocab)) if i not in all_indices]

    extracted_indices = sum([len(v) for v in indices.values()])
    assert extracted_indices == len(vocab), f"ERROR! extracted {extracted_indices} from {len(vocab)} indices"

    def stringify(_list):
        if len(_list) == 0:
            return f"---"
        elif len(_list) == 1:
            return f"{_list[0]} (#={len(_list)})"
        elif len(_list) == _list[-1] - _list[0] + 1:
            return f"{_list[0]}-{_list[-1]} (#={len(_list)})"
        else:
            return "ERROR!!"

    indices_str = {
        key: stringify(indices[key])
        for key in indices.keys()
    }

    print()
    print("=== overview vocabulary ===")
    for k, v in indices_str.items():
        print(f"{k}: {v}")

    return indices


def extract_vocab(_model):

    sp = spm.SentencePieceProcessor()
    sp.Load(_model)
    vocabs = {sp.IdToPiece(_id): _id for _id in range(sp.GetPieceSize())}

    if DEBUG:
        print("\n===")
        print(f"vocabulary size = {len(vocabs)}")
        print()
        print(f"examples:")
        for _id in range(5):
            print(sp.IdToPiece(_id), vocabs[sp.IdToPiece(_id)])
        for _id in range(len(vocabs)-5, len(vocabs)):
            print(sp.IdToPiece(_id), vocabs[sp.IdToPiece(_id)])
        print()

    # m = model.ModelProto()
    # m.ParseFromString(open(checkpoint, 'rb').read())
    # print(m.pieces)

    vocab_file = _model.replace("model.model", "tokenizer_vocab.json")
    if isfile(vocab_file):
        print(f"> vocabulary file {vocab_file} already exists.")
        print(f"SKIPPED.")
    else:
        print(f"> write vocabulary to {vocab_file}")
        with open(vocab_file, "w") as f:
            json.dump(vocabs, f)
        print(f"DONE.")
