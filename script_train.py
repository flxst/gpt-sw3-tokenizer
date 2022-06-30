"""
EXECUTION: python script_train.py
           --library
           --dataset_files
           --dataset_name
           --add_prefix_space
           --individual_digits
           --unicode_normalization
           --add_whitespace_tokens
           --minimum_frequency
           --vocab_size
           --alpha

PURPOSE: the script uses <library> to train a tokenizer named <tokenizer_name> on the <dataset_files>
         using the rest of the arguments as parameters.

         the trained tokenizer is saved at output/HHMMSS_[parameters]_<tokenizer_name>
"""
import argparse
from os.path import join
import time
from datasets import load_dataset
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from src.parameters import Parameters
from src.helpers import get_normalizer, get_training_corpus_combined
from src.output import Output

import sentencepiece as spm


def train_hf(_parameters, _output, _datasets_combined):
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


def train_sp(_parameters, _output, _datasets_combined):
    spm.SentencePieceTrainer.train(
        sentence_iterator=get_training_corpus_combined(_datasets_combined, batch_size=1),
        model_prefix=_output.model_prefix,
        model_type="BPE",
        bos_id=-1,
        eos_id=-1,
        # allow_whitespace_only_pieces=True,  # default: False
        # shrinking_factor=0.95,  # default: 0.75
        max_sentence_length=1000000,  # default: 4192, TODO
        normalization_rule_name="identity",                                     # 1. unicode normalization
        split_digits=_parameters.individual_digits,                             # 2. individual digits
        add_dummy_prefix=_parameters.add_prefix_space,                          # 3. add prefix space
        remove_extra_whitespaces=False,                                         # 4a. add whitespace
        user_defined_symbols=_parameters.special_tokens,                        # 4a. add whitespace & 4b. code tokens
        byte_fallback=_parameters.byte_fallback,                                # SP extra
        character_coverage=_parameters.character_coverage,                      # SP extra
        train_extremely_large_corpus=_parameters.train_extremely_large_corpus,  # SP extra
        vocab_size=_parameters.vocab_size,                                      # 6. vocabulary size
        minloglevel=1,  # default: 0 (=log everything?)
    )


def main(args):
    ts = time.time()

    # -1. Preparations: parameters & output
    parameters = Parameters(**vars(args))
    parameters.show()

    output = Output(parameters.output_dir, parameters.library)
    output.export_parameters(parameters)

    # 0. Load Datasets
    # if args.alpha != 1.0:
    #     upsampled_data_file = upsampling(parameters.dataset_files, args.alpha)
    #     data_files += upsampled_data_file
    datasets_combined = load_dataset('json', data_files={'train': parameters.dataset_files})

    if parameters.library == "HF":
        train_hf(parameters, output, datasets_combined)
    elif parameters.library == "SP":
        train_sp(parameters, output, datasets_combined)
    else:
        raise Exception(f"library = {parameters.library} unknown, should be HF or SP")

    # 5. Output
    output.export_tokenizer_for_megatron_lm()
    output.analyze_vocabulary()
    output.overview(datasets_combined, parameters.dataset_files, _time=f"{time.time() - ts:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=str, default="HF")
    parser.add_argument("--dataset_files", nargs='+', type=str, default=["data/test.json"])
    parser.add_argument("--dataset_name", type=str, default="?")
    parser.add_argument("--unicode_normalization", type=str, default="NFC")
    parser.add_argument("--individual_digits", type=int, default=1)
    parser.add_argument("--add_prefix_space", type=int, default=1)
    parser.add_argument("--add_whitespace_tokens", type=int, default=1)
    parser.add_argument("--add_code_tokens", type=int, default=1)
    parser.add_argument("--minimum_frequency", type=int, default=0)
    parser.add_argument("--byte_fallback", type=int, default=1)
    parser.add_argument("--character_coverage", type=float, default=1.0)
    parser.add_argument("--vocab_size", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=-1)
    parser.add_argument("--train_extremely_large_corpus", type=int, default=1)
    _args = parser.parse_args()

    main(_args)
