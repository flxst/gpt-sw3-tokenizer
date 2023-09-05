"""
EXECUTION: python script_train.py
           --tokenizer_name <tokenizer_name>      # e.g. tokenizer1
           --dataset_files <dataset_files>        # e.g. "all" = all files in <data_train>
           [--dataset_filter all]                 # e.g. "all" = no filter
           [--monolingual]                        # if used, monolingual models are trained
           [--library SP]                         # SP = SentencePiece, HF = HuggingFace
           [--unicode_normalization None]         # None, NFC, NFKC
           [--individual_digits 1]                # 0, 1
           [--add_prefix_space 1]                 # 0, 1
           [--add_whitespace_tokens 2]            # 0, 1 (added at top of vocab), 2 (added at bottom of vocab)
           [--add_code_tokens 1]                  # 0, 1 (added at top of vocab)
           [--minimum_frequency 0]                # int >= 0
           [--byte_fallback 1]                    # 0, 1
           [--character_coverage 0.9999]          # float, useful if byte_fallback = 1
           [--vocab_size 64000]                   # int, divisible by 128
           [--alpha -1]                           # upsampling parameter, -1 or 0 <= alpha < 1
           [--train_extremely_large_corpus 1]     # 0, 1

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
from os.path import join
import time
from typing import Union
from datasets import load_dataset, Dataset, DatasetDict, IterableDatasetDict, IterableDataset
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from src.parameters import Parameters
from src.helpers import get_normalizer, get_training_corpus_combined, get_languages
from src.helpers import add_special_tokens, create_merge_rules
from src.output import Output

import sentencepiece as spm
HFDataset = Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]


def train_hf(_parameters: Parameters,
             _output: Output,
             _datasets_combined: HFDataset) -> None:
    """
    train a HF tokenizer and produces the following output files in the tokenizer directory:
    - tokenizer.json

    Args:
        _parameters: for training
        _output: for model_prefix
        _datasets_combined: data to train on
    """

    # 1. Define Tokenizer
    tokenizer = Tokenizer(models.BPE())
    _normalizer = get_normalizer(_parameters.unicode_normalization)
    if _normalizer is not None:
        tokenizer.normalizer = _normalizer

    pre_tokenizer_features = [pre_tokenizers.ByteLevel(add_prefix_space=_parameters.add_prefix_space)]
    if _parameters.individual_digits:
        pre_tokenizer_features += [pre_tokenizers.Digits(individual_digits=True)]
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(pre_tokenizer_features)

    # 2. Train
    trainer = trainers.BpeTrainer(
        vocab_size=_parameters.vocab_size,
        special_tokens=_parameters.special_tokens,
        min_frequency=_parameters.minimum_frequency,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        # https://github.com/huggingface/tokenizers/issues/813#issuecomment-937847770
    )
    tokenizer.train_from_iterator(
        get_training_corpus_combined(_datasets_combined),
        trainer=trainer
    )

    # 3. Post-Processing
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()

    # 4. Save
    tokenizer.save(join(_output.path, "tokenizer.json"))


def train_sp(_parameters: Parameters,
             _output: Output,
             _datasets_combined: HFDataset) -> None:
    """
    train a SP tokenizer and produces the following output files in the tokenizer directory:
    - model.model
    - model.vocab

    Args:
        _parameters: for training
        _output: for model_prefix
        _datasets_combined: data to train on
    """
    spm.SentencePieceTrainer.train(
        sentence_iterator=get_training_corpus_combined(_datasets_combined, batch_size=1),
        model_prefix=_output.model_prefix,
        model_type="BPE",
        pad_id=0,   # previously: -1
        unk_id=1,   # previously: 0
        bos_id=2,   # previously: -1
        eos_id=3,   # previously: -1
        # pad_piece="<pad>",  # previously: not used, default: "<pad>"
        # unk_piece="<unk>",  # previously: "<unk>",  default: "<unk>"
        # bos_piece="<s>",    # previously: not used, default: "<s>"
        eos_piece="<|endoftext|>",   # previously: not used, default: "</s>"
        max_sentence_length=2000000,                                            # default: 4192
        normalization_rule_name="identity",                                     # 1. unicode normalization
        split_digits=_parameters.individual_digits,                             # 2. individual digits
        add_dummy_prefix=_parameters.add_prefix_space,                          # 3. add prefix space
        remove_extra_whitespaces=False,                                         # 4a. add whitespace
        user_defined_symbols=_parameters.special_tokens,                        # 4a. add whitespace & 4b. code tokens
        byte_fallback=_parameters.byte_fallback,                                # SP extra
        character_coverage=_parameters.character_coverage,                      # SP extra
        train_extremely_large_corpus=_parameters.train_extremely_large_corpus,  # SP extra
        vocab_size=_parameters.vocab_size,                                      # 6. vocabulary size
        minloglevel=1,                                                          # default: 0 (=log everything?)
    )

    if _parameters.add_whitespace_tokens == 2:
        add_special_tokens(_model_path=_output.path)


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
    parser.add_argument("--monolingual", action="store_true")
    parser.add_argument("--library", type=str, default="SP")
    parser.add_argument("--unicode_normalization", type=str, default="None")
    parser.add_argument("--individual_digits", type=int, default=1)
    parser.add_argument("--add_prefix_space", type=int, default=1)
    parser.add_argument("--add_whitespace_tokens", type=int, default=2)
    parser.add_argument("--add_code_tokens", type=int, default=1)
    parser.add_argument("--minimum_frequency", type=int, default=0)
    parser.add_argument("--byte_fallback", type=int, default=1)
    parser.add_argument("--character_coverage", type=float, default=0.9999)
    parser.add_argument("--vocab_size", type=int, default=64000)
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
