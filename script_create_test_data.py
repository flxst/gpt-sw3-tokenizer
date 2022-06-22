"""
EXECUTION: python script_create_test_data.py

PURPOSE: the script creates data files for testing:
         - data/test.json (contains TEST_CORPUS)
         - data/code.json (contains script_train.py as string)
"""
import json
from src.test_data import TEST_CORPUS


def main():
    test_data_file = "data/test.json"
    with open(test_data_file, "w", encoding="utf-8") as f:
        for i in range(len(TEST_CORPUS)):
            f.write(json.dumps({"text": TEST_CORPUS[i]}) + "\n")
    print(f"> wrote file '{test_data_file}'")

    with open("script_train.py", "r") as f:
        code = f.read().replace("\\n", "\n")
    print(code)

    code_data_file = "data/code.json"
    with open(code_data_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"text": code}))
    print(f"> wrote file '{code_data_file}'")


if __name__ == "__main__":
    main()
