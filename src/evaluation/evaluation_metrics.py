"""Module that contains class EvaluationMetrics which contains evaluation metrics"""
import sys
from typing import Dict, Any


class EvaluationMetrics:
    """Class which contains evaluation metrics"""

    def __init__(self):
        self.unk_rate = None
        self.ctcl = None
        self.fertility = None
        self.proportion = None
        self.token_frequencies = None

    def set(self, attribute: str, values: Dict[str, Any]) -> None:
        """
        set class attribute to value defined (directly or indirectly) in values

        Args:
            attribute: e.g. 'fertility'
            values: e.g. {'nominator': 100, 'denominator': 50}
        """
        # parse values
        if list(values.keys()) == ["value"]:
            value = values["value"]
        elif list(values.keys()) == ["nominator", "denominator"]:
            try:
                value = float(values["nominator"]) / float(values["denominator"])
            except ZeroDivisionError:
                value = -1
        else:
            sys.exit(
                f"ERROR! values needs to have keys == ['value'] or ['nominator', 'denominator']. "
                f"Got {values.keys()} instead."
            )

        # set attribute
        self.__dict__[attribute] = value

    def as_dict(self) -> Dict[str, Any]:
        """
        Returns:
            attributes_dict: class attributes as dict, e.g. {'unk_rate': 0.0, 'fertility': 2.0}
        """
        return self.__dict__
