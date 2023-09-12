from os.path import join
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

from src.helpers import get_normalizer, get_training_corpus_combined
from src.parameters import Parameters
from src.output import Output
from src.training.training import HFDataset


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


