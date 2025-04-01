import DataGenerator
import LinearRegression
from matplotlib import pyplot as plt


def LinearWithNoise(noise: float) -> None:
    fig = plt.figure(figsize=(8, 6))

    # Create scatter plot for linear function plus noise:
    (xData, yData) = DataGenerator.LinearGenerator(100, noise)
    p1 = plt.scatter(xData, yData)

    # Get optimal solutions for the line:
    (linearConstant, offSet) = LinearRegression.GetOptimalSolution(xData, yData)
    print(linearConstant, offSet)

    # Add to plot
    regressionOutput = linearConstant * xData + offSet
    plt.plot(xData, regressionOutput, "r")

    # Print mean squared error and its derivative
    print(LinearRegression.MeanSqrdError(linearConstant, offSet, xData, yData))

    print(
        LinearRegression.MeanSqrdErrorDerivative(linearConstant, offSet, xData, yData)
    )
