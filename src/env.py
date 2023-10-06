"""Module that contains the Env class to represent the environment"""
from os.path import join, abspath, dirname, isfile
import configparser

config = configparser.ConfigParser()
BASE_DIR = abspath(dirname(dirname(__file__)))


class Env:
    """Class used to represent the environment (data folders)"""

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
            relative_paths = False
        else:
            for field in ["data_train", "data_eval", "output"]:
                assert not config["main"][field].startswith(
                    "/"
                ), "ERROR! mixture of absolute and relative paths encountered in Env"
            relative_paths = True

        if relative_paths:
            self.data_original = join(BASE_DIR, config["main"]["data_original"])
            self.data_train = join(BASE_DIR, config["main"]["data_train"])
            self.data_eval = join(BASE_DIR, config["main"]["data_eval"])
            self.output = join(BASE_DIR, config["main"]["output"])
        else:
            self.data_original = config["main"]["data_original"]
            self.data_train = config["main"]["data_train"]
            self.data_eval = config["main"]["data_eval"]
            self.output = config["main"]["output"]

        self.debug = bool(int(config["other"]["debug"]))
        self.verbose = bool(int(config["other"]["verbose"]))


if __name__ == "__main__":
    env = Env()
    print(env.__dict__)
