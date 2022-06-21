from tokenizers import normalizers
from datasets import Dataset

UNICODE_NORMALIZATION = {
    "NFC": normalizers.NFC(),
    "NFKC": normalizers.NFKC(),
    "NFKD": normalizers.NFKD(),
}


def get_normalizer(_unicode_normalization: str):
    return UNICODE_NORMALIZATION[_unicode_normalization]


def get_training_corpus_combined(_dataset: Dataset, batch_size: int = 100000):
    for i in range(0, len(_dataset['train']), batch_size):
        yield _dataset['train'][i: i + batch_size]["text"]
