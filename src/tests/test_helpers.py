import pytest
from tokenizers.normalizers import Normalizer
from src.helpers import get_normalizer


class TestHelpers:
    @pytest.mark.parametrize(
        "unicode_normalization, is_normalizer",
        [
            ("None", False),
            ("NFC", True),
        ],
    )
    def test_get_normalizer(self, unicode_normalization: str, is_normalizer: bool):
        test_normalizer = get_normalizer(unicode_normalization)
        if is_normalizer is False:
            assert (
                test_normalizer is None
            ), f"ERROR! test_normalizer = {test_normalizer} != None"
        else:
            assert isinstance(
                test_normalizer, Normalizer
            ), f"ERROR! test_normalizer is not a Normalizer instance."
