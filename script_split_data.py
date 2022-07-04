
from os.path import join
import json
import argparse


def main(_args):
    max_sentence_length = _args.max_sentence_length
    original_path = join("data", "books_sv_epub_100.jsonl")
    new_path = original_path.replace(".jsonl", f"_max{max_sentence_length}.jsonl")

    print("\n=== ORIGINAL DATA ===")
    with open(original_path, "r", encoding="utf-8") as file:
        _data = [json.loads(line)["text"] for line in file]

    _lenghts = [len(elem) for elem in _data]
    print(f"{len(_lenghts)} docs, min/max/total characters = {min(_lenghts)}/{max(_lenghts)}/{sum(_lenghts)}")

    print("\n=== NEW DATA ===")
    _data_new = list()
    for elem in _data:
        for i in range(int(len(elem)/max_sentence_length) + 1):
            chunk = elem[i*max_sentence_length: (i+1)*max_sentence_length]
            _data_new.append(chunk)

    _lenghts = [len(elem) for elem in _data_new]
    print(f"{len(_lenghts)} docs, min/max/total characters = {min(_lenghts)}/{max(_lenghts)}/{sum(_lenghts)}")

    with open(new_path, "w", encoding="utf-8") as file:
        for elem in _data_new:
            file.write(json.dumps({"text": elem}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_sentence_length", type=int, required=True)
    _args = parser.parse_args()

    main(_args)
