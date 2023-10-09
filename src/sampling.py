"""Module that contains functions for sampling"""
import random
from typing import Tuple, List


def reservoir_sampling(
    infile, number_of_sampled_documents: int, exclude: Tuple[int, ...] = ()
) -> Tuple[List[str], Tuple[int, ...]]:
    """see
    https://stackoverflow.com/questions/40144869/python-read-random-lines-from-a-very-big-file-and-append-to-another-file

    Args:
        infile: file handler for opened ("r") file
        number_of_sampled_documents: e.g. 100
        exclude: tuple of excluded lines e.g. (1, 4, 5, )

    Returns:
        sample: sampled documents as single string, e.g.
        [
            '{"text": "this is test article number 4"}\n',
            '{"text": "this is test article number 2"}\n',
        ]
        sample_indices: e.g. (4, 2)
    """
    iteration = iter(infile)  # n = nr. of original documents, e.g. 5

    # create reservoir of k documents
    sample: List[str] = []
    sample_indices: List[int] = []
    count = {
        "iteration": 0,
        "skipped": 0,
    }
    while len(sample) < number_of_sampled_documents:
        if count["iteration"] not in exclude:
            try:
                sample.append(next(iteration))  # k = nr. of sampled documents, e.g. 2
                sample_indices.append(count["iteration"])
            except StopIteration as exc:
                raise ValueError("Sample larger than population") from exc
        else:
            _ = next(iteration)
            count["skipped"] += 1
        count["iteration"] += 1

    assert (
        len(sample) == number_of_sampled_documents
    ), f"ERROR! len(sample) = {len(sample)} != {number_of_sampled_documents} = number_of_sampled_documents"

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
                sample[random_int] = item
                sample_indices[random_int] = count["iteration"]
        else:
            count["skipped"] += 1
        count["iteration"] += 1

    # random.shuffle(sample)  # additional cost without effect
    return sample, tuple(sample_indices)


def reservoir_sampling_original(infile, number_of_sampled_documents: int) -> List[str]:
    """used only as a reference
    taken from
    https://stackoverflow.com/questions/40144869/python-read-random-lines-from-a-very-big-file-and-append-to-another-file
    """
    iteration = iter(infile)
    try:
        sample = [
            next(iteration) for _ in range(number_of_sampled_documents)
        ]  # use xrange if on python 2.x
    except StopIteration as exc:
        raise ValueError("Sample larger than population") from exc

    for i, item in enumerate(iteration, start=number_of_sampled_documents):
        random_int = random.randint(0, i)
        if random_int < number_of_sampled_documents:
            sample[random_int] = item

    # random.shuffle(sample)  # additional cost without effect
    return sample
