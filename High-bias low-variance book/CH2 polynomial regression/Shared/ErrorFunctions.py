import numpy as np


def MeanSqrdError(
    data: np.ndarray[float],
    expectedData: np.ndarray[float],
) -> float:
    assert len(data) == len(expectedData)
    length = len(expectedData)
    difVec = data - expectedData
    return np.dot(difVec, difVec) / length
