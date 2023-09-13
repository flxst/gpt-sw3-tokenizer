from typing import Optional
from os.path import join
from typing import Dict


class Logger:

    def __init__(self, logger_folder: str):
        self.log_file_path = join(logger_folder, "SAMPLING.log")

    def initialize(self, percent: str, weights: Dict[str, Dict[str, float]]):
        with open(self.log_file_path, "w") as f:
            f.write("======================\n")
            f.write(f"> PERCENT = {percent}\n")
            f.write(f"> WEIGHTS:\n")
            for category, language_dict in weights.items():
                for language, weight in language_dict.items():
                    f.write(f"  {category}, {language}: {weight}\n")
            f.write("======================\n\n")

    def log_print(self, _str: Optional[str] = None, end: Optional[str] = None):
        if _str is None:
            print()
            with open(self.log_file_path, "a") as f:
                f.write("\n")
        else:
            if end is None:
                print(_str)
                with open(self.log_file_path, "a") as f:
                    f.write(f"{_str}\n")
            else:
                print(_str, end=end)
                with open(self.log_file_path, "a") as f:
                    f.write(f"{_str}")
