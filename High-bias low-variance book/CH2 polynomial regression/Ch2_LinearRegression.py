from Shared import DataGenerator
from Shared import LinearRegression
from matplotlib import pyplot as plt


def LinearWithNoise(noise: float) -> None:
    fig = plt.figure(figsize=(8, 6))

    # Create scatter plot for linear function plus noise:
    (xData, yData) = DataGenerator.LinearGenerator(100, noise)
    scatterPlot = plt.scatter(xData, yData)

    # Get optimal solutions for the line and add to plot
    modelParameters = LinearRegression.GetOptimalSolution(xData, yData)
    linearConstant, offSet = modelParameters
    print(linearConstant, offSet)
    regressionOutput = linearConstant * xData + offSet
    fitPlot = plt.plot(xData, regressionOutput, "r")

    # Print mean squared error and its derivative
    print(LinearRegression.MeanSqrdError(modelParameters, xData, yData))
    print(LinearRegression.MeanSqrdErrorDerivative(modelParameters, xData, yData))
