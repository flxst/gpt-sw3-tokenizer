
from os.path import join
from src.helpers import create_merge_rules
from src.env import Env


if __name__ == "__main__":
    env = Env()
    # model_name = "104214_HF-uNone-d1-p1-w1-c1-f0-bf0-cc0-x0-v10000_2"
    # model_name = "151609_HF-uNone-d1-p1-w1-c1-f0-bf0-cc0-x0-v10000_2"
    # model_name = "151628_HF-uNone-d1-p1-w0-c1-f0-bf0-cc0-x0-v10000_2"
    model_name = "163843_HF-uNone-d1-p1-w0-c1-f0-bf0-cc0-x0-v20000_4da"

    vocab_file = join(env.output, model_name, "tokenizer_vocab.json")
    merge_file = join(env.output, model_name, "tokenizer_merge.txt")

    _ = create_merge_rules(vocab_file, merge_file)

