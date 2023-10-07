import pytest
from typing import Dict, Union, Optional, List
from os.path import join

from src.env import Env
from src.tests.helpers import BASE_DIR, TEST_DATA_DIRECTORY_RELATIVE

env = Env(TEST_DATA_DIRECTORY_RELATIVE)


class TestEnv:
    @pytest.mark.parametrize(
        "attributes",
        [
            (
                {
                    "data_original": join(
                        BASE_DIR, TEST_DATA_DIRECTORY_RELATIVE, "test_data_original"
                    ),
                    "data_train": join(
                        BASE_DIR, TEST_DATA_DIRECTORY_RELATIVE, "test_data_train"
                    ),
                    "data_eval": join(
                        BASE_DIR, TEST_DATA_DIRECTORY_RELATIVE, "test_data_eval"
                    ),
                    "output": join(
                        BASE_DIR, TEST_DATA_DIRECTORY_RELATIVE, "test_output"
                    ),
                    "sampling_weights": join(
                        BASE_DIR,
                        TEST_DATA_DIRECTORY_RELATIVE,
                        "test_SAMPLING_WEIGHTS.csv",
                    ),
                    "debug": False,
                    "verbose": False,
                }
            ),
        ],
    )
    def test_init(self, attributes: Dict[str, Union[str, bool]]):
        for key, value in attributes.items():
            assert attributes[key] == env.__getattribute__(
                key
            ), f"ERROR! test_attribute[{key}] = {env.__getattribute__(key)} != {attributes[key]}"

    @pytest.mark.parametrize(
        "category, language, kind, file_path",
        [
            (
                "books",
                "en",
                "data_original",
                join(
                    BASE_DIR,
                    TEST_DATA_DIRECTORY_RELATIVE,
                    "test_data_original",
                    "books_en.jsonl",
                ),
            ),
            (
                "books",
                "en",
                "data_train",
                join(
                    BASE_DIR,
                    TEST_DATA_DIRECTORY_RELATIVE,
                    "test_data_train",
                    "books_en.jsonl",
                ),
            ),
            (
                "books",
                "en",
                "data_eval",
                join(
                    BASE_DIR,
                    TEST_DATA_DIRECTORY_RELATIVE,
                    "test_data_eval",
                    "books_en.jsonl",
                ),
            ),
            ("books", "en", "data_not_existent", None),
        ],
    )
    def test_get_file_path(
        self, category: str, language: str, kind: str, file_path: Optional[str]
    ):
        if file_path is None:
            with pytest.raises(Exception):
                _ = env.get_file_path(category, language, kind)
        else:
            test_file_path = env.get_file_path(category, language, kind)
            assert test_file_path == file_path, (
                f"ERROR! test_file_path = {test_file_path} != {file_path} "
                f"for category = {category}, language = {language}, kind = {kind}"
            )

    @pytest.mark.parametrize(
        "percent, categories, languages, sampling_weights, sampling_weights_final",
        [
            (
                50,
                ["articles", "books"],
                ["en"],
                {"books": {"en": 0.5}, "articles": {"en": 1.0}},
                {"books": {"en": 0.25}, "articles": {"en": 0.5}},
            ),
        ],
    )
    def test_read_sampling_weights(
        self,
        percent: int,
        categories: List[str],
        languages: List[str],
        sampling_weights: Dict[str, Dict[str, float]],
        sampling_weights_final: Dict[str, Dict[str, float]],
    ):
        (
            test_categories,
            test_languages,
            test_sampling_weights,
            test_sampling_weights_final,
        ) = env.read_sampling_weights(percent, verbose=True)
        assert (
            test_categories == categories
        ), f"ERROR! test_categories = {test_categories} != {categories}"
        assert (
            test_languages == languages
        ), f"ERROR! test_languages = {test_languages} != {languages}"
        assert (
            test_sampling_weights == sampling_weights
        ), f"ERROR! test_sampling_weights = {test_sampling_weights} != {sampling_weights}"
        assert (
            test_sampling_weights_final == sampling_weights_final
        ), f"ERROR! test_sampling_weights_final = {test_sampling_weights_final} != {sampling_weights_final}"
