import numpy as np


def mid_frequencies(
    divider: int,
    n: np.array,
):
    """Calculates mid frequency for octave bands.

    Args:
        divider: [1,inf[, e.g. n=3 -> f_mid_n = 10**(n/10)
        n: array with the integers to calculate mid frequencies for, e.g.
            array([1,2,3,4,5])

    Returns:
        An array with mid frequencies.
    """
    exact_frequencies = 10**(3n/divider/10)


def lower_frequencies(
    divider: int,
    mid_frequencies: np.array,
):
    return mid_frequencies*10**(-3/divider/20) 


def upper_frequencies(
    divider: int,
    mid_frequencies: np.array,
):
    return mid_frequencies*10**(3/divider/20) 




