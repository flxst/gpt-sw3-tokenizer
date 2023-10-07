"""Module that contains the Env class to represent the environment"""
from os.path import join, abspath, dirname, isfile
import configparser
import csv
from typing import Tuple, List, Dict

config = configparser.ConfigParser()
BASE_DIR = abspath(dirname(dirname(__file__)))


class Env:
    """Class used to represent the environment (data folders & sampling weights file)"""

    def __init__(self, folder: str = "."):
        """
        Args:
            folder: the folder relative to the working directory where the environment file env.ini can be found
        """
        env_file = join(BASE_DIR, folder, "env.ini")
        assert isfile(env_file), f"ERROR! could not find file = {env_file}"

        config.read(env_file)

        if config["main"]["data_original"].startswith("/"):
            for field in ["data_train", "data_eval", "output"]:
                assert config["main"][field].startswith(
                    "/"
                ), "ERROR! mixture of absolute and relative paths encountered in Env"
            assert config["sampling"]["weights"].startswith(
                "/"
            ), "ERROR! mixture of absolute and relative paths encountered in Env"

            relative_paths = False
        else:
            for field in ["data_train", "data_eval", "output"]:
                assert not config["main"][field].startswith(
                    "/"
                ), "ERROR! mixture of absolute and relative paths encountered in Env"
            assert not config["sampling"]["weights"].startswith(
                "/"
            ), "ERROR! mixture of absolute and relative paths encountered in Env"
            relative_paths = True

        if relative_paths:
            self.data_original = join(BASE_DIR, config["main"]["data_original"])
            self.data_train = join(BASE_DIR, config["main"]["data_train"])
            self.data_eval = join(BASE_DIR, config["main"]["data_eval"])
            self.output = join(BASE_DIR, config["main"]["output"])
            self.sampling_weights = join(BASE_DIR, config["sampling"]["weights"])
        else:
            self.data_original = config["main"]["data_original"]
            self.data_train = config["main"]["data_train"]
            self.data_eval = config["main"]["data_eval"]
            self.output = config["main"]["output"]
            self.sampling_weights = config["sampling"]["weights"]

        self.debug = bool(int(config["other"]["debug"]))
        self.verbose = bool(int(config["other"]["verbose"]))

        assert isfile(
            self.sampling_weights
        ), f"ERROR! sampling weights file = {self.sampling_weights} not found."

    def get_file_path(self, category: str, language: str, kind: str) -> str:
        """
        get file path for data of kind 'kind', category 'category' and language 'language'

        Args:
            category: e.g. 'books'
            language: e.g. 'en'
            kind: e.g. 'data_original'

        Returns:
            file_path: e.g. '<data_original>/books_en.jsonl
        """
        if kind == "data_original":
            directory = self.data_original
        elif kind == "data_train":
            directory = self.data_train
        elif kind == "data_eval":
            directory = self.data_eval
        else:
            raise Exception(f"ERROR! kind = {kind} unknown.")

        return join(directory, f"{category}_{language}.jsonl")

    def read_sampling_weights(
        self, percent: int = 100, verbose: bool = False
    ) -> Tuple[
        List[str], List[str], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]
    ]:
        """
        read sampling weights from SAMPLING_WEIGHTS.csv and weight them globally with 'percent'

        Args:
            percent: e.g. 50
            verbose: e.g. False

        Returns:
            categories: e.g. ['articles', 'books']
            languages: e.g. ['en']
            sampling_weights: e.g. {'books': {'en': 0.5}, 'articles': {'en': 1.0}}
            sampling_weights_final: e.g. {'books': {'en': 0.25}, 'articles': {'en': 0.5}}
        """
        categories = []
        languages = []
        sampling_weights = {}
        with open(self.sampling_weights, "r", encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for j, row in enumerate(csv_reader):
                if j == 0:
                    languages = row[1:]
                else:
                    categories.append(row[0])
                    sampling_weights[row[0]] = {
                        languages[i - 1]: float(row[i])
                        for i in range(1, len(languages) + 1)
                    }

        sampling_weights_final = {
            _category: {_language: v2 * percent / 100 for _language, v2 in v1.items()}
            for _category, v1 in sampling_weights.items()
        }

        if verbose:
            print("\n> read sampling weights from SAMPLING_WEIGHTS.csv")
            print(f"  categories: {categories}")
            print(f"  languages: {languages}")
            print(f"  sampling_weights: {sampling_weights}")
            print(f"  sampling_weights_final: {sampling_weights_final}")

        return categories, languages, sampling_weights, sampling_weights_final


if __name__ == "__main__":
    env = Env()
    print(env.__dict__)
