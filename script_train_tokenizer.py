import argparse
from os.path import join
from os import makedirs
import time
from datasets import load_dataset, Dataset
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from src.parameters import Parameters
from src.helpers import get_normalizer, export_tokenizer_for_megatron_lm, analyze_vocabulary, overview


def get_training_corpus_combined(_dataset: Dataset, batch_size: int = 100000):
    for i in range(0, len(_dataset['train']), batch_size):
        yield _dataset['train'][i: i + batch_size]["text"]


def main(args):
    ts = time.time()
    parameters = Parameters(
        add_prefix_space=bool(args.add_prefix_space),
        individual_digits=bool(args.individual_digits),
        unicode_normalization=args.unicode_normalization,
        minimum_frequency=args.minimum_frequency,
        vocab_size=args.vocab_size,
        add_whitespace_tokens=args.add_whitespace_tokens,
        alpha=args.alpha,
    )

    data_files = args.input
    # if args.alpha != 1.0:
    #     upsampled_data_file = upsampling(data_files, args.alpha)
    #     data_files += upsampled_data_file

    datasets = list()

    output_dir = join("output", time.strftime("%H%M%S", time.localtime()))
    if parameters.use_id:
        suffix = f"_{args.dataset}-a{args.alpha}" if args.alpha != -1 else f"_{args.dataset}"
        output_dir += parameters.get_id() + suffix
    makedirs(output_dir, exist_ok=False)
    tokenizer_file = join(output_dir, "tokenizer")

    parameters.show()
    parameters.export(tokenizer_file)

    # 0. Load Datasets
    datasets_combined = load_dataset('json',
                                     data_files={'train': data_files},
                                     )

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
    tokenizer.train_from_iterator(get_training_corpus_combined(datasets_combined), trainer=trainer)

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
    parser.add_argument("--alpha", type=float, default=-1)
    _args = parser.parse_args()

    main(_args)
