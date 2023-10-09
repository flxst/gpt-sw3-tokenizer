"""Module that contains functions for sampling"""
import random
from typing import Tuple, List


def reservoir_sampling(
    infile, number_of_sampled_documents: int, exclude: Tuple[int] = ()
) -> List[str]:
    """see
    https://stackoverflow.com/questions/40144869/python-read-random-lines-from-a-very-big-file-and-append-to-another-file

    Args:
        infile: file handler for opened ("r") file
        number_of_sampled_documents: e.g. 100
        exclude: tuple of excluded lines e.g. (1, 4, 5, )

    Returns:
        sampled_documents as single string: e.g.
        [
            '{"text": "this is test article number 4"}\n',
            '{"text": "this is test article number 2"}\n',
        ]
    """
    iteration = iter(infile)  # n = nr. of original documents, e.g. 5

    # create reservoir of k documents
    result = []
    count = {
        "iteration": 0,
        "skipped": 0,
    }
    while len(result) < number_of_sampled_documents:
        if count["iteration"] not in exclude:
            try:
                result.append(next(iteration))  # k = nr. of sampled documents, e.g. 2
            except StopIteration as exc:
                raise ValueError("Sample larger than population") from exc
        else:
            _ = next(iteration)
            count["skipped"] += 1
        count["iteration"] += 1

    assert (
        len(result) == number_of_sampled_documents
    ), f"ERROR! len(result) = {len(result)} != {number_of_sampled_documents} = number_of_sampled_documents"

    # replace elements in reservoir
    for item in iteration:  # e.g. i = 2, 3, 4
        i = count["iteration"] - count["skipped"]
        if count["iteration"] not in exclude:
            random_int = random.randint(0, i)  # (i+1) possible numbers
            if random_int < number_of_sampled_documents:
                # probability = k / (i+1)
                # e.g. i = 2 => probability = 2/3 = k / (k + 1)
                # e.g. i = 3 => probability = 2/4
                # e.g. i = 4 => probability = 2/5 = k / n
                result[random_int] = item
        else:
            count["skipped"] += 1
        count["iteration"] += 1

    # random.shuffle(result)  # additional cost without effect
    return result


def reservoir_sampling_original(infile, number_of_sampled_documents: int) -> List[str]:
    """used only as a reference
    taken from
    https://stackoverflow.com/questions/40144869/python-read-random-lines-from-a-very-big-file-and-append-to-another-file
    """
    iteration = iter(infile)
    try:
        result = [
            next(iteration) for _ in range(number_of_sampled_documents)
        ]  # use xrange if on python 2.x
    except StopIteration as exc:
        raise ValueError("Sample larger than population") from exc

    for i, item in enumerate(iteration, start=number_of_sampled_documents):
        random_int = random.randint(0, i)
        if random_int < number_of_sampled_documents:
            result[random_int] = item

    # random.shuffle(result)  # additional cost without effect
    return result
