import pytest
from typing import List
from os.path import join
import random

from src.sampling import reservoir_sampling
from src.tests.helpers import BASE_DIR


class TestSampling:
    @pytest.mark.parametrize(
        "input_file, number_of_sampled_documents, random_seed, output_lines",
        [
            (
                "articles_en.jsonl",
                2,
                42,
                [
                    '{"text": "this is test article number 4"}\n',
                    '{"text": "this is test article number 2"}\n',
                ],
            ),
            (
                "articles_en.jsonl",
                2,
                43,
                [
                    '{"text": "this is test article number 3"}\n',
                    '{"text": "this is test article number 2"}\n',
                ],
            ),
            (
                "articles_en.jsonl",
                3,
                43,
                [
                    '{"text": "this is test article number 4"}\n',
                    '{"text": "this is test article number 2"}\n',
                    '{"text": "this is test article number 3"}\n',
                ],
            ),
            (
                "articles_en.jsonl",
                5,
                43,
                None,
            ),
        ],
    )
    def test_reservoir_sampling(
        self,
        input_file: str,
        number_of_sampled_documents: int,
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
                    _ = reservoir_sampling(infile, number_of_sampled_documents)
        else:
            with open(input_file_path, "r") as infile:
                test_output_lines = reservoir_sampling(
                    infile, number_of_sampled_documents
                )

            # test type = list
            assert isinstance(
                test_output_lines, list
            ), f"ERROR! type(test_output_lines) = {type(test_output_lines)} expected to be list"

            # test length
            assert (
                len(test_output_lines) == number_of_sampled_documents
            ), f"ERROR! len(test_output_lines) = {len(test_output_lines)} != {number_of_sampled_documents} = #sampled_docs"
            assert len(test_output_lines) == len(
                output_lines
            ), f"ERROR! len(test_output_lines) = {len(test_output_lines)} != {len(output_lines)} = len(output_lines)"

            # test single documents
            for i, (test_output_line, output_line) in enumerate(
                zip(test_output_lines, output_lines)
            ):
                assert (
                    test_output_line == output_line
                ), f"ERROR! line #{i} is different: {test_output_line} vs. {output_line}"
