import os
import subprocess
from os.path import abspath, isdir, join, isfile
import shutil


def print_section_header(header: str) -> None:
    print(f"========================================================================")
    print(f"=== {header} ")


def print_section_finish() -> None:
    print(f"=== SUCCESS\n")


def run_cli(bash_cmd: str) -> None:
    try:
        result = subprocess.run(bash_cmd, shell=True, check=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
        print(result)
    except subprocess.CalledProcessError as e:
        raise Exception(e)


def clear_directory(_directory_path: str) -> None:
    if isdir(_directory_path):
        shutil.rmtree(_directory_path)
        print(f"> removed {_directory_path}\n")
    os.makedirs(_directory_path)


def check_e2e_data_original() -> None:
    e2e_data_original_directory = abspath("./e2e_data_original")
    assert isdir(e2e_data_original_directory), f"ERROR! directory {e2e_data_original_directory} does not exist."
    for file in [
        "articles_en.jsonl",
        "books_hq_en.jsonl",
        "conversational_en.jsonl",
        "math_en.jsonl",
        "misc_en.jsonl",
        "web_commoncrawl_en.jsonl",
        "web_sources_en.jsonl",
    ]:
        file_path = join(e2e_data_original_directory, file)
        assert isfile(file_path), f"ERROR! file {file_path} does not exist."


def test_e2e(capsys):

    # check existence of e2e_data_original
    check_e2e_data_original()

    # clear directories from previous runs
    data_dir = abspath("./e2e_tests/e2e_test_data")
    e2e_dirs = [
        abspath("./e2e_data_train"),
        abspath("./e2e_data_eval"),
        abspath("./e2e_data_eval"),
        abspath("./e2e_output")
    ]
    for _directory_path in [data_dir] + e2e_dirs:
        clear_directory(_directory_path)

    # end-to-end test
    try:
        ################################################################################################################
        print_section_header(f"1a. sampling (train)")
        run_cli("python script_sampling.py --percent 50")
        print_section_finish()

        ################################################################################################################
        print_section_header(f"1b. sampling (eval)")
        run_cli("python script_sampling.py --percent 20 --evaluation 1")
        print_section_finish()

        ################################################################################################################
        print_section_header(f"2. train")
        run_cli("python script_train.py --tokenizer_name tokenizer_test --dataset_files all")
        print_section_finish()

        ################################################################################################################
        print_section_header(f"3. evaluation")
        run_cli("python script_evaluate.py --tokenizer_name tokenizer_test")
        print_section_finish()
    except Exception as e:
        raise Exception(e)
    finally:
        # stdout & stderr to files
        out, err = capsys.readouterr()
        with open(join(data_dir, "err.txt"), "w") as f:
            f.write(err)
        with open(join(data_dir, "out.txt"), "w") as f:
            f.write(out)
