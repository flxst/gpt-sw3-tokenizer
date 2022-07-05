
import os
from os.path import join, abspath
import configparser

config = configparser.ConfigParser()


class Env:

    def __init__(self,
                 folder="."):
        cwd = os.getcwd()
        base_dir = abspath(join(cwd, folder))
        env_file = join(base_dir, "env.ini")
        config.read(env_file)

        self.data_original = join(base_dir, config['main']['data_original'])
        self.data_sampled = join(base_dir, config['main']['data_sampled'])
        self.output = join(base_dir, config['main']['output'])


if __name__ == "__main__":
    env = Env()
    print(env.__dict__)
