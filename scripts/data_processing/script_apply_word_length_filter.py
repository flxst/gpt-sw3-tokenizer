"""
EXECUTION: python script_apply_word_length_filter.py
           --directory <directory>
           [--threshold 20000]

PURPOSE: the script
         - takes all the dataset files in <directory>
         - for each dataset file, checks whether a document has a non-whitespace sequence of length > <threshold>
         - if so, it filters those documents and writes the rest to <directory>_FILTERED
"""
import os
from os.path import join, isdir, isfile
import json
import argparse

from os.path import abspath, dirname
import sys
BASE_DIR = abspath(dirname(dirname(dirname(abspath(__file__)))))
print(f">>> BASE_DIR: {BASE_DIR}")
sys.path.append(BASE_DIR)


def main(args):

    input_directory = join(BASE_DIR, args.directory)
    output_directory = join(BASE_DIR, f"{input_directory}_FILTERED")

    assert isdir(input_directory), f"ERROR! {input_directory} not found."

    input_files = [
        join(input_directory, elem)
        for elem in os.listdir(input_directory)
        if isfile(join(input_directory, elem)) and "_FILTERED.jsonl" not in elem
    ]

    print()
    print(f"> found {len(input_files)} files in {input_directory}")

    for i, input_file in enumerate(input_files):
        print(f"\n=== file #{i+1}: {input_file}")
        output_file = join(output_directory, input_file.split("/")[-1].replace(".jsonl", "_FILTERED.jsonl"))

        # read old data
        with open(input_file, "r", encoding="utf-8") as _file:
            data = [json.loads(line) for line in _file]
        print(f"> loaded {len(data)} documents from {input_file}")

        # new data: apply word length filter
        new_data = list()
        counter = {'kept': 0, 'removed': 0}
        for document in data:
            longest_word = max([len(elem) for elem in document["text"].split(" ")])
            if longest_word <= args.threshold:
                new_data.append(document)
                counter['kept'] += 1
            else:
                counter['removed'] += 1
        print(f"> applied filter with threshold = {args.threshold}")
        print(f"  kept    documents: {counter['kept']}")
        print(f"  removed documents: {counter['removed']}")

        # write new data if needed
        if counter['removed'] > 0:
            os.makedirs(output_directory, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as _file:
                for line in new_data:
                    _file.write(json.dumps(line, ensure_ascii=False) + "\n")
            print(f"> wrote {len(new_data)} documents to {output_file}")
        else:
            print(f"> no need to write a new file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, required=True)
    parser.add_argument("--threshold", type=int, default=20000)
    _args = parser.parse_args()

    main(_args)
