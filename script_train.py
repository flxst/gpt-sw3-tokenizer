"""
EXECUTION: python script_train.py
           --dataset_files
           --dataset_name
           --add_prefix_space
           --individual_digits
           --unicode_normalization
           --add_whitespace_tokens
           --minimum_frequency
           --vocab_size
           --alpha

PURPOSE: the script trains a tokenizer named <tokenizer_name> on the <dataset_files>
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


def main(args):
    ts = time.time()

    # -1. Preparations: parameters & output
    parameters = Parameters(**vars(args))
    parameters.show()

    output = Output(parameters.output_dir)
    output.export_parameters(parameters)

    # 0. Load Datasets
    # if args.alpha != 1.0:
    #     upsampled_data_file = upsampling(parameters.dataset_files, args.alpha)
    #     data_files += upsampled_data_file
    datasets_combined = load_dataset('json', data_files={'train': parameters.dataset_files})

    # 1. Define Tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token=parameters.unk_token))
    tokenizer.normalizer = get_normalizer(parameters.unicode_normalization)
    pre_tokenizer_features = [pre_tokenizers.ByteLevel(add_prefix_space=parameters.add_prefix_space)]
    if parameters.individual_digits:
        pre_tokenizer_features += [pre_tokenizers.Digits(individual_digits=True)]
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(pre_tokenizer_features)

    # 2. Train
    trainer = trainers.BpeTrainer(
        vocab_size=parameters.vocab_size,
        special_tokens=parameters.special_tokens,
        min_frequency=parameters.minimum_frequency,
    )
    tokenizer.train_from_iterator(
        get_training_corpus_combined(datasets_combined),
        trainer=trainer
    )

    # 3. Post-Processing
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()

    # 4. Save
    tokenizer.save(join(output.path, "tokenizer.json"))

    # 5. Output
    output.export_tokenizer_for_megatron_lm()
    output.analyze_vocabulary()
    output.overview(datasets_combined, parameters.dataset_files, _time=f"{time.time() - ts:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_files", nargs='+', type=str, default=["data/test.json"])
    parser.add_argument("--dataset_name", type=str, default="?")
    parser.add_argument("--unicode_normalization", type=str, default="NFC")
    parser.add_argument("--individual_digits", type=int, default=1)
    parser.add_argument("--add_prefix_space", type=int, default=1)
    parser.add_argument("--add_whitespace_tokens", type=int, default=24)
    parser.add_argument("--minimum_frequency", type=int, default=0)
    parser.add_argument("--vocab_size", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=-1)
    _args = parser.parse_args()

    main(_args)
