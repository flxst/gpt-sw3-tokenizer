from typing import Union
from datasets import Dataset, DatasetDict, IterableDatasetDict, IterableDataset

HFDataset = Dataset  # Union[Dataset, DatasetDict, IterableDatasetDict, IterableDataset]
