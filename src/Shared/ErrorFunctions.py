from .CustomTyping import *
import numpy as np


def MeanSqrdError(
    data: npFloatArray,
    expectedData: npFloatArray,
) -> float:
    assert len(data) == len(expectedData)
    length = len(expectedData)
    difVec = data - expectedData
    return np.dot(difVec, difVec) / length
