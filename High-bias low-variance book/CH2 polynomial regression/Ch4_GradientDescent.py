from Shared import NewtonsMethod
from Shared import LinearRegression
import numpy as np


def GenerateCurriedErrorFunctions(
    yData: np.ndarray[float],
) -> tuple[callable, callable]:
    def curriedError(xData: np.ndarray[float], parameters: np.ndarray[float]):
        return LinearRegression.MeanSqrdError(parameters[0], parameters[1])


def BasicNewtonsMethods():
    fig = plt.figure(figsize=(8, 6))

    # Create scatter plot for linear function plus noise:
    (xData, yData) = DataGenerator.LinearGenerator(100, noise)
    scatterPlot = plt.scatter(xData, yData)

    iterator = NewtonsMethod()
