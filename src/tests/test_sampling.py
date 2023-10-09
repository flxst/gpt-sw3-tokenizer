import pytest
from typing import List, Tuple
from os.path import join
import random

from src.sampling import reservoir_sampling, reservoir_sampling_original
from src.tests.helpers import BASE_DIR


class TestSampling:
    @pytest.mark.parametrize(
        "input_file, number_of_sampled_documents, exclude, random_seed, output_lines",
        [
            # 1
            (
                "articles_en.jsonl",
                2,
                (),
                42,
                [
                    '{"text": "this is test article number 3"}\n',
                    '{"text": "this is test article number 1"}\n',
                ],
            ),
            # 2
            (
                "articles_en.jsonl",
                2,
                (),
                43,
                [
                    '{"text": "this is test article number 2"}\n',
                    '{"text": "this is test article number 1"}\n',
                ],
            ),
            # 3
            (
                "articles_en.jsonl",
                3,
                (),
                43,
                [
                    '{"text": "this is test article number 3"}\n',
                    '{"text": "this is test article number 1"}\n',
                    '{"text": "this is test article number 2"}\n',
                ],
            ),
            # 4
            (
                "articles_en.jsonl",
                5,
                (),
                43,
                None,
            ),
            # 5
            (
                "articles_en.jsonl",
                2,
                (1,),
                42,
                [
                    '{"text": "this is test article number 0"}\n',
                    '{"text": "this is test article number 2"}\n',
                ],
            ),
            # 6
            (
                "articles_en.jsonl",
                3,
                (3,),
                43,
                [
                    '{"text": "this is test article number 0"}\n',
                    '{"text": "this is test article number 1"}\n',
                    '{"text": "this is test article number 2"}\n',
                ],
            ),
            # 7
            (
                "articles_en.jsonl",
                3,
                (1,),
                43,
                [
                    '{"text": "this is test article number 0"}\n',
                    '{"text": "this is test article number 2"}\n',
                    '{"text": "this is test article number 3"}\n',
                ],
            ),
            # 8
            (
                "articles_en.jsonl",
                4,
                (1,),
                43,
                None,
            ),
        ],
    )
    def test_reservoir_sampling(
        self,
        input_file: str,
        number_of_sampled_documents: int,
        exclude: Tuple[int],
        random_seed: int,
        output_lines: List[str],
    ):
        random.seed(random_seed)
        input_file_path = join(
            BASE_DIR, "src", "tests", "data", "test_data_original", input_file
        )

        if output_lines is None:
            with pytest.raises(Exception):
                with open(input_file_path, "r") as infile:
                    _ = reservoir_sampling(infile, number_of_sampled_documents, exclude)
        else:
            with open(input_file_path, "r") as infile:
                test_output_lines = reservoir_sampling(
                    infile, number_of_sampled_documents, exclude
                )
            _test_sampling(test_output_lines, number_of_sampled_documents, output_lines)

    @pytest.mark.parametrize(
        "input_file, number_of_sampled_documents, exclude, random_seed, output_lines",
        [
            # 1
            (
                    "articles_en.jsonl",
                    2,
                    (),
                    42,
                    [
                        '{"text": "this is test article number 3"}\n',
                        '{"text": "this is test article number 1"}\n',
                    ],
            ),
            # 2
            (
                    "articles_en.jsonl",
                    2,
                    (),
                    43,
                    [
                        '{"text": "this is test article number 2"}\n',
                        '{"text": "this is test article number 1"}\n',
                    ],
            ),
            # 3
            (
                    "articles_en.jsonl",
                    3,
                    (),
                    43,
                    [
                        '{"text": "this is test article number 3"}\n',
                        '{"text": "this is test article number 1"}\n',
                        '{"text": "this is test article number 2"}\n',
                    ],
            ),
            # 4
            (
                    "articles_en.jsonl",
                    5,
                    (),
                    43,
                    None,
            ),
        ],
    )
    def test_reservoir_sampling_original(
            self,
            input_file: str,
            number_of_sampled_documents: int,
            exclude: Tuple[int],  # not used
            random_seed: int,
            output_lines: List[str],
    ):
        random.seed(random_seed)
        input_file_path = join(
            BASE_DIR, "src", "tests", "data", "test_data_original", input_file
        )

        if output_lines is None:
            with pytest.raises(Exception):
                with open(input_file_path, "r") as infile:
                    _ = reservoir_sampling_original(infile, number_of_sampled_documents)
        else:
            with open(input_file_path, "r") as infile:
                test_output_lines = reservoir_sampling_original(
                    infile, number_of_sampled_documents
                )
                _test_sampling(test_output_lines, number_of_sampled_documents, output_lines)


def _test_sampling(_test_output_lines, _number_of_sampled_documents, _output_lines):
    # test type = list
    assert isinstance(
        _test_output_lines, list
    ), f"ERROR! type(test_output_lines) = {type(_test_output_lines)} expected to be list"

    # test length
    assert (
            len(_test_output_lines) == _number_of_sampled_documents
    ), f"ERROR! len(test_output_lines) = {len(_test_output_lines)} != {_number_of_sampled_documents} = #sampled_docs"
    assert len(_test_output_lines) == len(
        _output_lines
    ), f"ERROR! len(test_output_lines) = {len(_test_output_lines)} != {len(_output_lines)} = len(output_lines)"

    # test single documents
    for i, (test_output_line, output_line) in enumerate(
            zip(_test_output_lines, _output_lines)
    ):
        assert (
                test_output_line == output_line
        ), f"ERROR! line #{i} is different: {test_output_line} vs. {output_line}"
