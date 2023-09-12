from typing import Union
from datasets import Dataset, DatasetDict, IterableDatasetDict, IterableDataset

HFDataset = Union[Dataset, DatasetDict, IterableDatasetDict, IterableDataset]
