"""Module that contains functions for sampling"""
import random


def reservoir_sampling(infile, number_of_sampled_documents: int):
    """
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
