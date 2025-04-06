import numpy as np

from ..custom_typing import npFloatArray
from .model import Model
from .mean_squared_error import MeanSqrdError


class LinearModel(Model):
    def __init__(self, parameters: npFloatArray) -> None:
        Model.__init__(
            self, LinearModelFunc, MeanSqrdError, MeanSqrdErrorDerivative, parameters
        )


def LinearModelFunc(parameters: npFloatArray, inputData: npFloatArray) -> npFloatArray:
    assert len(parameters) == 2
    return parameters[0] + parameters[1] * inputData


# TODO: this does not work, needs info from parameters
# assumes inputdata is calculated using the linear model function
def MeanSqrdErrorDerivative(
    parameters: npFloatArray,
    inputData: npFloatArray,
    expectedOutput: npFloatArray,
) -> npFloatArray:
    assert len(inputData) == len(expectedOutput)
    length = len(expectedOutput)
    difVec = LinearModelFunc(parameters, inputData) - expectedOutput
    p0Deriv = 2 * np.sum(difVec) / length
    p1Deriv = 2 * np.dot(inputData, difVec) / length
    return np.array([p0Deriv, p1Deriv])
