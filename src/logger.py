"""Module that contains the Logger class to print and log to file"""
from typing import Optional, Dict
from os.path import join


class Logger:
    """Class used to print and log to file"""

    def __init__(self, logger_folder: str):
        self.log_file_path = join(logger_folder, "SAMPLING.log")

    def initialize(
        self, percent: str, sampling_weights: Dict[str, Dict[str, float]]
    ) -> None:
        """
        logs initial information about the sampling process

        Args:
            percent: e.g. '10'
            sampling_weights: e.g. {'books': {'en': 0.5}, 'articles': {'en': 1.0}}
        """
        with open(self.log_file_path, "w", encoding="utf-8") as file:
            file.write("======================\n")
            file.write(f"> PERCENT = {percent}\n")
            file.write("> WEIGHTS:\n")
            for category, language_dict in sampling_weights.items():
                for language, weight in language_dict.items():
                    file.write(f"  {category}, {language}: {weight}\n")
            file.write("======================\n\n")

    def log_print(self, _str: Optional[str] = None, end: Optional[str] = None) -> None:
        """
        print (to stdout) and logs (to file) the input string '_str'

        Args:
            _str: input str, e.g. 'this is a test'
            end: e.g. '', which avoids a line break
        """
        if _str is None:
            print()
            with open(self.log_file_path, "a", encoding="utf-8") as file:
                file.write("\n")
        else:
            if end is None:
                print(_str)
                with open(self.log_file_path, "a", encoding="utf-8") as file:
                    file.write(f"{_str}\n")
            else:
                print(_str, end=end)
                with open(self.log_file_path, "a", encoding="utf-8") as file:
                    file.write(f"{_str}")
