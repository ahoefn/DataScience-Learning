from shared.gradient_descent import newtons_method
from shared import linear_regression
from shared import data_generator
from shared.custom_typing import npFloatArray
import numpy as np
from matplotlib import pyplot as plt


class Ch4_GradientDescent:
    def __init__(self, noise: float) -> None:
        self.fig = plt.figure(figsize=(8, 6))

        # Create scatter plot for linear function plus noise:
        (xData, yData) = data_generator.LinearGenerator(100, noise)
        self.scatterPlot = plt.scatter(xData, yData)

        initialParameters: npFloatArray = np.random.randn(2)

        self.iterator = newtons_method.NewtonOptimizer(
            linear_regression.LinearModelFunc,
            linear_regression.MeanSqrdError,
            linear_regression.MeanSqrdErrorDerivative,
            initialParameters,
            xData,
            yData,
            0.1,
        )
        self.errorLog: list[float] = [self.iterator.GetCurrentError()]

    def StepAndLog(self) -> None:
        self.iterator.step()
        self.errorLog.append(self.iterator.GetCurrentError())


if __name__ == "__main__":
    newton = Ch4_GradientDescent(0.5)

    newton.scatterPlot = plt.scatter(
        newton.iterator.inputData, newton.iterator.expectedResults
    )
    for i in range(2000):
        newton.StepAndLog()

        plt.plot(newton.iterator.inputData, newton.iterator.GetCurrentOutput())

    fig = plt.figure(figsize=(8, 6))
    errorPlot = plt.plot(newton.errorLog)
    plt.show()
    print(newton.iterator.parameters)
