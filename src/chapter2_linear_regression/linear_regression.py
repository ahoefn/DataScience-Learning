from shared.custom_typing import npFloatArray
import numpy as np


def LinearModelFunc(parameters: npFloatArray, inputData: npFloatArray) -> npFloatArray:
    return parameters[0] * inputData + parameters[1]


def MeanSqrdError(
    parameters: npFloatArray,
    inputData: npFloatArray,
    expectedOutput: npFloatArray,
) -> float:
    assert len(inputData) == len(expectedOutput)
    length = len(expectedOutput)
    difVec = LinearModelFunc(parameters, inputData) - expectedOutput
    return np.dot(difVec, difVec) / length


def MeanSqrdErrorDerivative(
    parameters: npFloatArray,
    inputData: npFloatArray,
    expectedOutput: npFloatArray,
) -> npFloatArray:
    assert len(inputData) == len(expectedOutput)
    length = len(expectedOutput)
    difVec = LinearModelFunc(parameters, inputData) - expectedOutput
    aDerivative = 2 * np.dot(inputData, difVec) / length
    bDerivative = 2 * np.sum(difVec) / length
    return np.array([aDerivative, bDerivative])


# Algebraically solved optimal choices for the constants
def GetOptimalSolution(
    inputData: npFloatArray,
    expectedOutput: npFloatArray,
) -> npFloatArray:
    assert len(inputData) == len(expectedOutput)
    length = len(expectedOutput)
    inputSum = np.sum(inputData)
    outputSum = np.sum(expectedOutput)
    linearOptimal = (
        length * np.dot(inputData, expectedOutput) - inputSum * outputSum
    ) / (length * np.dot(inputData, inputData) - np.square(inputSum))

    offSetOptimal = (outputSum - linearOptimal * inputSum) / length
    return np.array([linearOptimal, offSetOptimal])
