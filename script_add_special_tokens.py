
import os
from os.path import join
from src.helpers import add_special_tokens
from src.env import Env


if __name__ == "__main__":
    env = Env()

    model_name = "142226_SP-uNone-d1-p1-w0-c1-f0-bf0-cc1.0-x1-v10000_2"
    model_path = join(env.output, "p_w", model_name)
    os.makedirs(join(env.output, "p_w"), exist_ok=True)
    add_special_tokens(model_path, overwrite=False)
