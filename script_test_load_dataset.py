from datasets import load_dataset, Dataset
from typing import List


def get_training_corpus(_datasets: List[Dataset], batch_size: int):
    for dataset in _datasets:
        for i in range(0, len(dataset['train']), batch_size):
            yield dataset['train'][i: i + batch_size]["text"]


def get_training_corpus_combined(_dataset: Dataset, batch_size: int):
    for i in range(0, len(_dataset['train']), batch_size):
        yield _dataset['train'][i: i + batch_size]["text"]


"""
def get_training_corpus_combined_nohf(_data_files: List[str]):
    for data_file in _data_files:
        with open(data_file, 'r') as lines:
            for line in lines:
                yield json.loads(line)['text']
"""

"""
from datasets import Features, Value, Sequence
features = Features({
    'text': Value(dtype='string', id=None),
    'filename': Value(dtype='string', id=None),
    'filters': Sequence(feature=Value(dtype='null', id=None), length=-1, id=None),
    'keep': Value(dtype='int64', id=None),
    'len_char': Value(dtype='int64', id=None),
    'len_utf8bytes': Value(dtype='int64', id=None),
    'len_words': Value(dtype='int64', id=None),
    'len_sents': Value(dtype='int64', id=None),
    'lang': Value(dtype='string', id=None),
    'md5': Value(dtype='string', id=None),
    'id': Value(dtype='string', id=None),
    'url': Value(dtype='string', id=None),
    'size_pdf': Value(dtype='int64', id=None),
})
"""

BATCH_SIZE = 100000
if 1:
    DATA_FILES = ["data/wiki_is.jsonl", "data/wiki_da.jsonl"]
else:
    DATA_FILES = ["data/data_100_final/conversational/final_is.jsonl",
                  "data/data_100_final/conversational/final_da.jsonl"]


def main():
    # 0. Load Datasets
    datasets = [
        load_dataset('json',
                     data_files={'train': data_file},
                     # field=["text", "id"],
                     # features=features,
                     )
        for data_file in DATA_FILES
    ]

    datasets_combined = load_dataset('json',
                                     data_files={'train': DATA_FILES},
                                     )

    for i, dataset in enumerate([datasets, datasets_combined]):
        print(f"\n=== {i} ===")
        print(f"> number of datasets: {len(dataset)}")

        if i == 0:
            for j in range(len(dataset)):
                print(f"> length dataset {j}: {len(dataset[j]['train'])}")
            generator = get_training_corpus(dataset, BATCH_SIZE)
        else:
            print(f"> length dataset: {len(dataset['train'])}")
            generator = get_training_corpus_combined(dataset, BATCH_SIZE)

        for elem in generator:
            print(type(elem), len(elem), len(elem[0]), elem[0][:10])


if __name__ == "__main__":
    main()
