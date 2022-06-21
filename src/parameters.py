import inspect
from typing import List


class Parameters:
    
    def __init__(self,
                 add_prefix_space: bool = True,  
                 individual_digits: bool = True,
                 unicode_normalization: str = "NFC",
                 minimum_frequency: int = 0,
                 vocab_size: int = 100,
                 add_whitespace_tokens: int = 0,
                 alpha: float = 1.0):

        # add_prefix_space: "Let .." becomes [('Let', (0, 3)), ..] if False, [('ĠLet', (0, 3)), ..] if True
        self.add_prefix_space = add_prefix_space
        self.individual_digits = individual_digits
        self.unicode_normalization = unicode_normalization
        self.minimum_frequency = minimum_frequency
        self.vocab_size = vocab_size
        self.add_whitespace_tokens = add_whitespace_tokens
        self.alpha = alpha

        # DERIVED
        self.unk_token: str = "[UNK]"
        self.special_tokens: List[str] = ["<|endoftext|>"] + [self.unk_token]

        # Experimental:
        self.whitespace_token = " "
        if self.add_whitespace_tokens > 0:
            whitespace_tokens = [
                self.whitespace_token * i
                for i in range(2, self.add_whitespace_tokens + 1)  # 2-24 consecutive whitespaces
            ]
            self.special_tokens += whitespace_tokens

        self.use_id: bool = True

    def show(self) -> None:
        """ print parameters """
        print("=== PARAMETERS ===")
        print(f"> add_prefix_space = {self.add_prefix_space}")
        print(f"> add_whitespace_tokens = {self.add_whitespace_tokens}")
        print(f"> individual_digits = {self.individual_digits}")
        print(f"> minimum_frequency = {self.minimum_frequency}")
        print(f"> special_tokens = {self.special_tokens}")
        print(f"> unicode_normalization = {self.unicode_normalization}")
        print(f"> unknown_token = {self.unk_token}")
        print(f"> vocab_size = {self.vocab_size}")
        print(f"> alpha = {self.alpha}")
        print("==================")
        print()

    def export(self, _tokenizer_file: str) -> None:
        """ export parameters to file

        Args:
            _tokenizer_file: e.g. 'output/125842/tokenizer => export file = 'output/125842/parameters.txt'
        """
        _parameters_file = f"{'/'.join(_tokenizer_file.split('/')[:-1])}/parameters.txt"
        with open(_parameters_file, "w") as f:
            for tup in inspect.getmembers(self):
                if not tup[0].startswith("__") and not callable(tup[1]):
                    print(tup)
                    f.write(f"{tup[0]} = {tup[1]}\n")

    def get_id(self) -> str:

        return "_" + \
            f"u{self.unicode_normalization}-" + \
            f"d{int(self.individual_digits)}-" + \
            f"p{int(self.add_prefix_space)}-" + \
            f"w{self.add_whitespace_tokens}-" + \
            f"f{self.minimum_frequency}-" + \
            f"v{self.vocab_size}"  # + f"v{self.vocab_size}-" + f"a{self.alpha}"
