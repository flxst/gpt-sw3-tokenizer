
from typing import Dict, Any


class EvaluationMetrics:

    def __init__(self):
        self.unk_rate = None
        self.ctcl = None
        self.fertility = None
        self.proportion = None
        self.token_frequencies = None

    def set(self, attribute: str, values: Dict[str, Any]) -> None:

        # parse values
        if list(values.keys()) == ["value"]:
            value = values["value"]
        elif list(values.keys()) == ["nominator", "denominator"]:
            try:
                value = float(values["nominator"]) / float(values["denominator"])
            except ZeroDivisionError:
                value = -1
        else:
            raise Exception(f"ERROR! values needs to have keys == ['value'] or ['nominator', 'denominator']. "
                            f"Got {values.keys()} instead.")

        # set attribute
        self.__dict__[attribute] = value

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__
