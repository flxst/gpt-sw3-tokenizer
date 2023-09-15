"""
EXECUTION: python script_train.py
           --tokenizer_name <tokenizer_name>      # e.g. tokenizer1
           --dataset_files <dataset_files>        # e.g. "all" = all files in <data_train>
           [--dataset_filter all]                 # e.g. "all" = no filter
           [--vocab_size 64000]                   # int, divisible by 128 (only SP)
           [--monolingual]                        # if used, monolingual models are trained
           [--library SP]                         # SP = SentencePiece, HF = HuggingFace
           [--unicode_normalization None]         # None, NFC, NFKC
           [--individual_digits 1]                # 0, 1
           [--add_prefix_space 1]                 # 0, 1
           [--add_whitespace_tokens 2]            # 0, 1 (added at top of vocab), 2 (added at bottom of vocab)
           [--add_code_tokens 1]                  # 0, 1 (added at top of vocab)
           [--add_newline_token 0]                # 0, 1 (added at top of vocab)
           [--minimum_frequency 0]                # int >= 0
           [--initial_alphabet 0]                 # 0, 1 (only HF)
           [--byte_fallback 1]                    # 0, 1 (only SP)
           [--character_coverage 0.9999]          # float, useful if byte_fallback = 1 (only SP)
           [--train_extremely_large_corpus 1]     # 0, 1 (only SP)

PURPOSE: the script uses <library> to train a tokenizer named <tokenizer_name> on the <dataset_files>
         using the rest of the arguments as parameters.

         the trained tokenizer is saved at <output>/YYmmdd_HHMMSS-v<vocab_size>_<tokenizer_name>
         and contains the following files:
         - Tokenizer: model.model & model.vocab (if library == "SP") or tokenizer.json (if library == "HF")
         - Training parameters: parameters.txt
         - Files for Megatron-LM: tokenizer_vocab.json & tokenizer_merge.txt
         - Tokenizer Statistics: tokenizer_subword_lengths.json
         - Dataset Statistics: overview.json
"""
import argparse
import time
from datasets import load_dataset

from src.parameters import Parameters
from src.helpers import get_languages
from src.output import Output
from src.training.training_hf import train_hf
from src.training.training_sp import train_sp


def main(args):
    ts = time.time()

    # -1. Preparations: parameters & output
    parameters = Parameters(**vars(args))
    parameters.show()

    output = Output(parameters.output_dir, parameters.library)
    output.export_parameters(parameters)  # output: parameters.txt

    # 0. Load Datasets
    # if args.alpha != 1.0:
    #     upsampled_data_file = upsampling(parameters.dataset_files, args.alpha)
    #     data_files += upsampled_data_file
    datasets_combined = load_dataset('json', data_files={'train': parameters.dataset_files})

    # 1. Train Tokenizer
    if parameters.library == "HF":
        train_hf(parameters, output, datasets_combined)  # output: tokenizer.json
    elif parameters.library == "SP":
        train_sp(parameters, output, datasets_combined)  # output: model.model & model.vocab
    else:
        raise Exception(f"library = {parameters.library} unknown, should be HF or SP")

    # 2. Output
    output.export_tokenizer_for_megatron_lm()  # output: tokenizer_vocab.json (+ tokenizer_merge.txt if library == HF)
    output.analyze_vocabulary()                # output: tokenizer_subword_lengths.json
    output.overview(                           # output: overview.json
        datasets_combined,
        parameters.dataset_files,
        _time=f"{time.time() - ts:.2f}s"
    )

    # if parameters.library == "SP":
    #     create_merge_rules(output.vocab_file, output.merge_file)  # output: tokenizer_merge.txt if library == SP


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", type=str, default="")
    parser.add_argument("--dataset_files", nargs='+', type=str, default=[])
    parser.add_argument("--dataset_filter", type=str, default="all")
    parser.add_argument("--vocab_size", type=int, default=64000)
    parser.add_argument("--monolingual", action="store_true")
    parser.add_argument("--library", type=str, default="SP")
    parser.add_argument("--unicode_normalization", type=str, default="None")
    parser.add_argument("--individual_digits", type=int, default=1)
    parser.add_argument("--add_prefix_space", type=int, default=1)
    parser.add_argument("--add_whitespace_tokens", type=int, default=2)
    parser.add_argument("--add_code_tokens", type=int, default=1)
    parser.add_argument("--add_newline_token", type=int, default=0)
    parser.add_argument("--minimum_frequency", type=int, default=0)
    parser.add_argument("--initial_alphabet", type=int, default=0)
    parser.add_argument("--byte_fallback", type=int, default=1)
    parser.add_argument("--character_coverage", type=float, default=0.9999)
    parser.add_argument("--alpha", type=float, default=-1)
    parser.add_argument("--train_extremely_large_corpus", type=int, default=1)
    _args = parser.parse_args()

    monolingual = _args.__dict__.pop("monolingual")
    if monolingual is False:
        main(_args)
    else:
        for language in get_languages("train"):
            _args.tokenizer_name += f"_{language}"
            _args.dataset_filter = f"_{language}"
            main(_args)
