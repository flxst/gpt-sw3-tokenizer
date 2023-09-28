import sentencepiece as spm

from src.helpers import get_training_corpus_combined, add_special_tokens
from src.parameters import Parameters
from src.output import Output
from src.training.training import HFDataset


def train_sp(
    _parameters: Parameters, _output: Output, _datasets_combined: HFDataset
) -> None:
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
        sentence_iterator=get_training_corpus_combined(
            _datasets_combined, batch_size=1
        ),
        model_prefix=_output.model_prefix,
        model_type="BPE",
        pad_id=0,  # previously: -1
        unk_id=1,  # previously: 0
        bos_id=2,  # previously: -1
        eos_id=3,  # previously: -1
        # pad_piece="<pad>",  # previously: not used, default: "<pad>"
        # unk_piece="<unk>",  # previously: "<unk>",  default: "<unk>"
        # bos_piece="<s>",    # previously: not used, default: "<s>"
        eos_piece="<|endoftext|>",  # previously: not used, default: "</s>"
        max_sentence_length=2000000,  # default: 4192
        normalization_rule_name="identity",  # 1. unicode normalization
        split_digits=_parameters.individual_digits,  # 2. individual digits
        add_dummy_prefix=_parameters.add_prefix_space,  # 3. add prefix space
        remove_extra_whitespaces=False,  # 4a. add whitespace
        user_defined_symbols=_parameters.special_tokens,  # 4a. add whitespace & 4b. code tokens
        byte_fallback=_parameters.byte_fallback,  # SP extra
        character_coverage=_parameters.character_coverage,  # SP extra
        train_extremely_large_corpus=_parameters.train_extremely_large_corpus,  # SP extra
        vocab_size=_parameters.vocab_size,  # 6. vocabulary size
        minloglevel=1,  # default: 0 (=log everything?)
    )

    if _parameters.add_whitespace_tokens == 2:
        add_special_tokens(_model_path=_output.path)
