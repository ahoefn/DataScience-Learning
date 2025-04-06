import numpy as np

from ..custom_typing import npFloatArray


def MeanSqrdError(
    inputData: npFloatArray,
    expectedOutput: npFloatArray,
) -> float:
    assert len(inputData) == len(expectedOutput)
    length = len(expectedOutput)
    difVec = inputData - expectedOutput
    return np.dot(difVec, difVec) / length
