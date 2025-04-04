from shared import data_generator
from shared import linear_regression
from matplotlib import pyplot as plt


def LinearWithNoise(noise: float) -> None:
    plt.figure(figsize=(8, 6))

    # Create scatter plot for linear function plus noise:
    (xData, yData) = data_generator.LinearGenerator(100, noise)
    plt.scatter(xData, yData)

    # Get optimal solutions for the line and add to plot
    modelParameters = linear_regression.GetOptimalSolution(xData, yData)
    linearConstant, offSet = modelParameters
    print(linearConstant, offSet)
    regressionOutput = linearConstant * xData + offSet
    plt.plot(xData, regressionOutput, "r")

    plt.show()

    # Print mean squared error and its derivative
    print(linear_regression.MeanSqrdError(modelParameters, xData, yData))
    print(linear_regression.MeanSqrdErrorDerivative(modelParameters, xData, yData))
