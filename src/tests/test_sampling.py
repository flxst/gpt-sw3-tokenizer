import pytest
from typing import List, Tuple
from os.path import join
import random

from src.sampling import reservoir_sampling, reservoir_sampling_original
from src.tests.helpers import BASE_DIR


class TestSampling:
    @pytest.mark.parametrize(
        "input_file, number_of_sampled_documents, exclude, random_seed, sample, sample_indices",
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
                (3, 1),
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
                (2, 1),
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
                (3, 1, 2),
            ),
            # 4
            (
                "articles_en.jsonl",
                5,
                (),
                43,
                None,
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
                (0, 2),
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
                (0, 1, 2),
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
                (0, 2, 3),
            ),
            # 8
            (
                "articles_en.jsonl",
                4,
                (1,),
                43,
                None,
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
        sample: List[str],
        sample_indices: Tuple[int, ...],
    ):
        random.seed(random_seed)
        input_file_path = join(
            BASE_DIR, "src", "tests", "data", "test_data_original", input_file
        )

        if sample is None:
            with pytest.raises(Exception):
                with open(input_file_path, "r") as infile:
                    _, _ = reservoir_sampling(
                        infile, number_of_sampled_documents, exclude
                    )
        else:
            with open(input_file_path, "r") as infile:
                test_sample, test_sample_indices = reservoir_sampling(
                    infile, number_of_sampled_documents, exclude
                )
            _test_sampling(
                test_sample,
                test_sample_indices,
                number_of_sampled_documents,
                sample,
                sample_indices,
            )

    @pytest.mark.parametrize(
        "input_file, number_of_sampled_documents, exclude, random_seed, sample, sample_indices",
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
                (3, 1),
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
                (2, 1),
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
                (3, 1, 2),
            ),
            # 4
            (
                "articles_en.jsonl",
                5,
                (),
                43,
                None,
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
        sample: List[str],
        sample_indices: Tuple[int, ...],  # not used
    ):
        random.seed(random_seed)
        input_file_path = join(
            BASE_DIR, "src", "tests", "data", "test_data_original", input_file
        )

        if sample is None:
            with pytest.raises(Exception):
                with open(input_file_path, "r") as infile:
                    _ = reservoir_sampling_original(infile, number_of_sampled_documents)
        else:
            with open(input_file_path, "r") as infile:
                test_sample = reservoir_sampling_original(
                    infile, number_of_sampled_documents
                )
                _test_sampling(
                    test_sample, None, number_of_sampled_documents, sample, None
                )


def _test_sampling(
    _test_sample,
    _test_sample_indices,
    _number_of_sampled_documents,
    _sample,
    _sample_indices,
):
    # test type = list
    assert isinstance(
        _test_sample, list
    ), f"ERROR! type(test_sample) = {type(_test_sample)} expected to be list"

    if _test_sample_indices is not None:
        assert isinstance(
            _test_sample_indices, tuple
        ), f"ERROR! type(test_sample_indices) = {type(_test_sample)} expected to be list"

    # test length
    assert (
        len(_test_sample) == _number_of_sampled_documents
    ), f"ERROR! len(test_sample) = {len(_test_sample)} != {_number_of_sampled_documents} = #sampled_docs"
    assert len(_test_sample) == len(
        _sample
    ), f"ERROR! len(test_sample) = {len(_test_sample)} != {len(_sample)} = len(sample)"

    if _test_sample_indices is not None:
        assert (
            len(_test_sample_indices) == _number_of_sampled_documents
        ), f"ERROR! len(test_sample_indices) = {len(_test_sample_indices)} != {_number_of_sampled_documents} = #sampled_docs"
        assert len(_test_sample_indices) == len(
            _sample
        ), f"ERROR! len(test_sample_indices) = {len(_test_sample_indices)} != {len(_sample)} = len(sample)"

    # test single documents
    for i, (test_elem, elem) in enumerate(zip(_test_sample, _sample)):
        assert (
            test_elem == elem
        ), f"ERROR! line #{i} is different: {test_elem} vs. {elem}"

    # test single indices
    if _test_sample_indices is not None:
        for i, (test_elem, elem) in enumerate(
            zip(_test_sample_indices, _sample_indices)
        ):
            assert (
                test_elem == elem
            ), f"ERROR! index #{i} is different: {test_elem} vs. {elem}"
