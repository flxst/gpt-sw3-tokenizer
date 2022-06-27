from typing import List
from os.path import join
import time


class Parameters:
    
    def __init__(self,
                 dataset_files: List[str],
                 dataset_name: str,
                 unicode_normalization: str = "NFC",
                 individual_digits: bool = True,
                 add_prefix_space: bool = True,
                 add_whitespace_tokens: bool = True,
                 minimum_frequency: int = 0,
                 vocab_size: int = 100,
                 alpha: float = 1.0):

        self.dataset_files = dataset_files
        self.dataset_name = dataset_name
        self.unicode_normalization = unicode_normalization
        self.individual_digits = bool(individual_digits)
        self.add_prefix_space = bool(add_prefix_space)
        self.add_whitespace_tokens = bool(add_whitespace_tokens)
        self.minimum_frequency = minimum_frequency
        self.vocab_size = vocab_size
        self.alpha = alpha

        # DERIVED
        self.special_tokens: List[str] = ["<|endoftext|>"]

        if self.add_whitespace_tokens:
            whitespace_token = " "
            whitespace_tokens = [
                whitespace_token * i
                for i in range(2, 25)  # 2-24 consecutive whitespaces
            ]
            self.special_tokens += whitespace_tokens

        suffix = f"_{self.dataset_name}-a{self.alpha}" if self.alpha != -1 else f"_{self.dataset_name}"
        self.output_dir = join("output", time.strftime("%H%M%S", time.localtime())) + self.get_id() + suffix

    def show(self) -> None:
        """ print parameters """
        print("=== PARAMETERS ===")
        print(f"> unicode_normalization = {self.unicode_normalization}")
        print(f"> individual_digits = {self.individual_digits}")
        print(f"> add_prefix_space = {self.add_prefix_space}")
        print(f"> add_whitespace_tokens = {self.add_whitespace_tokens}")
        print(f"> minimum_frequency = {self.minimum_frequency}")
        print(f"> vocab_size = {self.vocab_size}")
        print(f"> alpha = {self.alpha}")
        print("==================")
        print(f"> special_tokens = {self.special_tokens}")
        print("==================")
        print()

    def get_id(self) -> str:

        return "_" + \
            f"u{self.unicode_normalization}-" + \
            f"d{int(self.individual_digits)}-" + \
            f"p{int(self.add_prefix_space)}-" + \
            f"w{int(self.add_whitespace_tokens)}-" + \
            f"f{self.minimum_frequency}-" + \
            f"v{self.vocab_size}"  # + f"v{self.vocab_size}-" + f"a{self.alpha}"
