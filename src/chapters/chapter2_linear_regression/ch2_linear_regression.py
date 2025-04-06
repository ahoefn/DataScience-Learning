from matplotlib import pyplot as plt

from chapters.shared import data_generator
from . import linear_regression


def Run() -> None:
    LinearWithNoise(0.2)


def LinearWithNoise(noise: float) -> None:
    plt.figure(figsize=(8, 6))

    # Create scatter plot for linear function plus noise:
    (xData, yData) = data_generator.LinearGenerator(100, noise)
    plt.scatter(xData, yData)

    # Get optimal solutions for the line and add to plot
    modelParameters = linear_regression.GetOptimalSolution(xData, yData)
    linearConstant, offSet = modelParameters
    print(f"Linear constant set to {linearConstant}, offset to {offSet}")
    regressionOutput = linearConstant * xData + offSet
    plt.plot(xData, regressionOutput, "r")

    # Print mean squared error and its derivative
    print(
        f"Mean Squared error is {linear_regression.MeanSqrdError(modelParameters, xData, yData)}"
    )
    print(
        f"Mean Squared error derivative is {linear_regression.MeanSqrdErrorDerivative(modelParameters, xData, yData)}"
    )

    plt.show()


if __name__ == "__main__":
    LinearWithNoise(0.2)
