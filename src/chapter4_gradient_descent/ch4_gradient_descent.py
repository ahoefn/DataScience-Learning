import numpy as np
from matplotlib import pyplot as plt

from shared.custom_typing import npFloatArray
from shared.gradient_descent import gradient_descent
from chapter2_linear_regression.linear_regression import GetOptimalSolution
from shared.models import LinearModel
from shared import data_generator


def Run() -> None:
    noise = 0.2

    plt.figure(figsize=(8, 6))

    # Create scatter plot for linear function plus noise:
    (xData, yData) = data_generator.LinearGenerator(200, noise)
    plt.scatter(xData, yData)

    # Construct iterator with linear model
    initialParameters: npFloatArray = np.random.randn(2)
    iterator = gradient_descent.NewtonOptimizer(
        LinearModel(initialParameters),
        xData,
        yData,
        0.1,
    )
    errorLog: list[float] = [iterator.GetCurrentCost()]

    # Start optimization:
    for i in range(5000):
        iterator.step()
        errorLog.append(iterator.GetCurrentCost())

        plt.plot(iterator.inputData, iterator.GetCurrentOutput())
        print(iterator.GetCurrentCost())

    plt.plot(iterator.inputData, iterator.GetCurrentOutput())
    plt.figure(figsize=(8, 6))
    plt.plot(errorLog)

    print(f"Output parameters: {iterator.model.parameters}")
    print(f"Final error is {iterator.GetCurrentCost()}")
    print(f"Final derivative is {iterator.GetCurrentDeriv()}")

    plt.show()
