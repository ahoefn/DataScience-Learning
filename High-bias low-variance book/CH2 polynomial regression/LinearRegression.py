import numpy as np


def MeanSqrdError(
    linearConstant: float,
    offSet: float,
    inputData: np.ndarray[float],
    expectedOutput: np.ndarray[float],
) -> float:
    assert len(inputData) == len(expectedOutput)
    length = len(expectedOutput)
    difVec = linearConstant * inputData + offSet - expectedOutput
    return np.dot(difVec, difVec) / length


def MeanSqrdErrorDerivative(
    linearConstant: float,
    offSet: float,
    inputData: np.ndarray[float],
    expectedOutput: np.ndarray[float],
) -> tuple[float, float]:
    assert len(inputData) == len(expectedOutput)
    length = len(expectedOutput)
    difVec = linearConstant * inputData + offSet - expectedOutput
    aDerivative = 2 / length * np.dot(inputData, difVec)
    bDerivative = 2 / length * np.sum(difVec)
    return (aDerivative, bDerivative)


# Algebraically solved optimal choices for the constants
def GetOptimalSolution(
    inputData: np.ndarray[float],
    expectedOutput: np.ndarray[float],
) -> tuple[float, float]:
    assert len(inputData) == len(expectedOutput)
    length = len(expectedOutput)
    inputSum = np.sum(inputData)
    outputSum = np.sum(expectedOutput)
    linearOptimal = (
        length * np.dot(inputData, expectedOutput) - inputSum * outputSum
    ) / (length * np.dot(inputData, inputData) - np.square(inputSum))

    offSetOptimal = (outputSum - linearOptimal * inputSum) / length
    return (linearOptimal, offSetOptimal)
