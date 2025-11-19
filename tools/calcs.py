import numpy as np

def moving_average(a, n=3):
    """calculate a centred moving average"""
    if n < 2:
        raise ValueError(
            "Window size (n) must be at least 2 for a centered moving average."
        )

    data = np.empty_like(a, dtype=float)
    data.fill(np.nan)

    # Calculate the cumulative sum
    cumsum = np.cumsum(np.insert(a, 0, 0))

    # Calculate the centered moving average
    half_n = n // 2
    if n % 2 == 0:
        data[half_n - 1 : -half_n] = (cumsum[n:] - cumsum[:-n]) / n
    else:
        data[half_n:-half_n] = (cumsum[n:] - cumsum[:-n]) / n

    return data