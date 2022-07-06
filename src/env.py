
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

        if config['main']['data_original'].startswith("/"):
            for field in ['data_sampled', 'data_eval', 'output']:
                assert config['main'][field].startswith("/"), \
                    f"ERROR! mixture of absolute and relative paths encountered in Env"
            relative_paths = False
        else:
            for field in ['data_sampled', 'data_eval', 'output']:
                assert not config['main'][field].startswith("/"), \
                    f"ERROR! mixture of absolute and relative paths encountered in Env"
            relative_paths = True

        if relative_paths:
            self.data_original = join(base_dir, config['main']['data_original'])
            self.data_sampled = join(base_dir, config['main']['data_sampled'])
            self.data_eval = join(base_dir, config['main']['data_eval'])
            self.output = join(base_dir, config['main']['output'])
        else:
            self.data_original = config['main']['data_original']
            self.data_sampled = config['main']['data_sampled']
            self.data_eval = config['main']['data_eval']
            self.output = config['main']['output']


if __name__ == "__main__":
    env = Env()
    print(env.__dict__)
