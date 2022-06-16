import argparse
from os.path import isfile, join
from os import makedirs
import time
from datasets import load_dataset
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from parameters import Parameters
from helpers import get_normalizer, export_tokenizer_for_megatron_lm, analyze_vocabulary, overview


def main(args):
    ts = time.time()
    parameters = Parameters(
        add_prefix_space=bool(args.add_prefix_space),
        individual_digits=bool(args.individual_digits),
        unicode_normalization=args.unicode_normalization,
        minimum_frequency=args.minimum_frequency,
        vocab_size=args.vocab_size,
        add_whitespace_tokens=args.add_whitespace_tokens,
    )

    data_files = args.input
    datasets = list()

    # output_dir = join("output", time.strftime("%Y-%d-%m___%H-%M-%S", time.localtime()))
    output_dir = join("output", time.strftime("%H%M%S", time.localtime()))
    if parameters.use_id:
        output_dir += parameters.get_id() + f"_{args.dataset}"
    makedirs(output_dir, exist_ok=False)
    tokenizer_file = join(output_dir, "tokenizer")

    parameters.show()
    parameters.export(tokenizer_file)

    # 0. Load Datasets
    for j, data_file in enumerate(data_files):
        assert isfile(data_file), f"ERROR! {data_file} does not exist."
        datasets.append(load_dataset('json', data_files={'train': data_file}))

    def get_training_corpus(_datasets):
        for dataset in datasets:
            for i in range(0, len(dataset), 1000):
                yield dataset['train'][i: i + 1000]["text"]

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
    tokenizer.train_from_iterator(get_training_corpus(datasets), trainer=trainer)

    # 3. Post-Processing
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()

    # 4. Save
    tokenizer.save(tokenizer_file + ".json")

    # 5. Export for Megatron-LM
    export_tokenizer_for_megatron_lm(tokenizer_file)

    # 6. Analyze vocabulary
    analyze_vocabulary(tokenizer_file)

    # 7. Overview
    te = time.time()
    tm = f"{te - ts:.2f}s"
    overview(tokenizer_file, datasets, data_files, tm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs='+', type=str, default=["data/test.json"])
    parser.add_argument("--dataset", type=str, default="?")
    parser.add_argument("--add_prefix_space", type=int, default=1)
    parser.add_argument("--individual_digits", type=int, default=1)
    parser.add_argument("--unicode_normalization", type=str, default="NFC")
    parser.add_argument("--add_whitespace_tokens", type=int, default=0)
    parser.add_argument("--minimum_frequency", type=int, default=0)
    parser.add_argument("--vocab_size", type=int, default=500)
    _args = parser.parse_args()

    main(_args)
