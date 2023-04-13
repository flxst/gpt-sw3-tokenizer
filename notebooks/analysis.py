
from typing import Dict, List

from src.env import Env
from src.analysis import _analyze_vocab


def analyze_vocab(_model) -> Dict[str, List[int]]:
    env = Env("..")

    return _analyze_vocab(env, _model)
