"""
EXECUTION: python script_concatenate_data_by_language.py
           --directory <directory>

PURPOSE: the script
         - takes all the dataset files in <directory>
         - merges them by language
         - writes the results to <directory>_CONCATENATED_BY_LANGUAGE
"""
import os
from os.path import join, isdir, isfile
import argparse
import shutil

from os.path import abspath, dirname
import sys
BASE_DIR = abspath(dirname(dirname(dirname(abspath(__file__)))))
print(f">>> BASE_DIR: {BASE_DIR}")
sys.path.append(BASE_DIR)


from src.helpers import LANGUAGES


def concatenate_data_by_language(directory, inplace=True):

    input_directory = join(BASE_DIR, directory)
    output_directory = input_directory if inplace else join(BASE_DIR, f"{input_directory}_CONCATENATED_BY_LANGUAGE")

    assert isdir(input_directory), f"ERROR! {input_directory} not found."

    input_files_all = [
        join(input_directory, elem)
        for elem in os.listdir(input_directory)
        if isfile(join(input_directory, elem))
        and "_CONCATENATED_BY_LANGUAGE.jsonl" not in elem
        and "all_" not in elem
    ]

    print()
    print(f"> found {len(input_files_all)} files")

    input_files_by_language = {
        lang: [input_file for input_file in input_files_all if f"_{lang}" in input_file.split("/")[-1]]
        for lang in LANGUAGES
    }
    for lang, input_files in input_files_by_language.items():
        print(f"  {lang}: {len(input_files)} files in {input_directory}")

    # 2. write to output_directory
    if inplace is False:
        os.makedirs(output_directory, exist_ok=False)
    print()
    for lang, input_files in input_files_by_language.items():
        output_file = join(output_directory, f"all_{lang}.jsonl")

        with open(output_file, 'w') as wfd:
            for f in input_files:
                with open(f, 'r') as fd:
                    shutil.copyfileobj(fd, wfd)

        print(f"> wrote {len(input_files)} files to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, required=True)
    _args = parser.parse_args()

    concatenate_data_by_language(directory=_args.directory, inplace=False)
