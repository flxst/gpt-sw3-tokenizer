
from os.path import join
from src.helpers import create_merge_rules


if __name__ == "__main__":
    # model_name = "093017_SP-uNone-d1-p1-w1-c1-f0-bf1-cc0.9999-x1-v128000_3da"
    model_name = "102249_SP-uNone-d1-p1-w2-c1-f0-bf1-cc0.9999-x1-v10000_2"

    vocab_file = join("output", model_name, "tokenizer_vocab.json")
    merge_file = join("output", model_name, "tokenizer_merge.txt")

    _ = create_merge_rules(vocab_file, merge_file)

