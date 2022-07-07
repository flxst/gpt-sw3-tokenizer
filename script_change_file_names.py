
import argparse
import os
from os.path import isfile, isdir, join


def main(args):
    directory = join(os.getcwd(), args.directory)
    assert isdir(directory), f"ERROR! directory = {directory} does not exist"
    assert (args.remove_percent and not args.add_percent) or (not args.remove_percent and args.add_percent), \
        f"ERROR! need to EITHER use --remove_percent ({args.remove_percent}) " \
        f"OR specify --add_percent <int> ({args.add_percent})"
    files = [join(directory, elem) for elem in os.listdir(directory) if isfile(join(directory, elem))]
    for file in files:
        if args.remove_percent:
            assert file.endswith(f"_{args.remove_percent}p.jsonl"), \
                f"ERROR! file = {file} does not end with _{args.remove_percent}p.jsonl"
            new_file = file.replace(f"_{args.remove_percent}p.jsonl", ".jsonl")
        elif args.add_percent:
            assert file.endswith(f".jsonl"), \
                f"ERROR! file = {file} does not end with .jsonl"
            new_file = file.replace(".jsonl", f"_{args.add_percent}p.jsonl")
        else:
            raise Exception("error.")
        os.rename(file, new_file)
        print(f"> {file} moved to {new_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, required=True)
    parser.add_argument("--remove_percent", type=int, default=0)
    parser.add_argument("--add_percent", type=int, default=0)
    _args = parser.parse_args()

    main(_args)
