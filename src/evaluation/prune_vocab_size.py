
import os
from os.path import join
from typing import List
from copy import deepcopy
from sentencepiece import sentencepiece_model_pb2 as model_pb2

from src.env import Env
from src.analysis import _analyze_vocab

env = Env()


def prune_vocab_size(_tokenizer: str,
                     _vocab_sizes: List[int],
                     _last_regular_token_index: int) -> List[str]:
    """only works for library == SP"""  # TODO
    _tokenizers = [_tokenizer]

    if len(_vocab_sizes) > 1:
        # initial tokenizer
        vocab_size_tokenizer = _vocab_sizes[-1]
        assert str(vocab_size_tokenizer) in _tokenizer, \
            f"ERROR! vocab size = {vocab_size_tokenizer} is not in _tokenizer = {_tokenizer}"
        tokenizer_file = join(_tokenizer, "model.model")
        m = model_pb2.ModelProto()
        m.ParseFromString(open(tokenizer_file, 'rb').read())

        indices = _analyze_vocab(env, _tokenizer)
        print()
        print(f"> last regular token index: {indices['merges'][-1]}")

        # pruned tokenizers
        for _vocab_size in _vocab_sizes[:-1]:
            pruned_tokenizer_dir = _tokenizer.replace(f"v{vocab_size_tokenizer}", f"v{_vocab_size}")
            os.makedirs(pruned_tokenizer_dir, exist_ok=False)
            pruned_tokenizer = join(pruned_tokenizer_dir, "model.model")

            m_pruned = deepcopy(m)
            for i, _ in enumerate(m.pieces):
                if _last_regular_token_index - (vocab_size_tokenizer - _vocab_size) < i <= _last_regular_token_index:
                    # workaround: overwrite with extremely unlikely token
                    m_pruned.pieces[i].piece = f"a!?x$$▁!!xyz.masdf_{i}"

            for j in [0, 1, 2, vocab_size_tokenizer-23, vocab_size_tokenizer-22]:
                assert m.pieces[j].piece == m_pruned.pieces[j].piece, \
                    f"ERROR for j = {j}, piece: {m.pieces[j].piece} != {m_pruned.pieces[j].piece}"
                assert m.pieces[j].score == m_pruned.pieces[j].score, \
                    f"ERROR for j = {j}, score: {m.pieces[j].score} != {m_pruned.pieces[j].score}"

            if env.debug:
                for j in [0, _vocab_size - 23, _vocab_size - 22, -24, -23, -1]:
                    print(j)
                    print(m.pieces[j])
                    print(m_pruned.pieces[j])

            with open(pruned_tokenizer, 'wb') as f:
                f.write(m_pruned.SerializeToString())
            print(f"> wrote new tokenizer to {pruned_tokenizer}")

            _tokenizers.append(pruned_tokenizer_dir)

    return _tokenizers
