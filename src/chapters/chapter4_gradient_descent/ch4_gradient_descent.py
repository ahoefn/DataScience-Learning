import numpy as np
from matplotlib import pyplot as plt

from chapters.shared.custom_typing import npFloatArray
from chapters.shared.gradient_descent import GradientDescent
from chapters.shared.models import LinearModel
from chapters.shared import data_generator


def Run() -> None:
    noise = 0.2

    plt.figure(figsize=(8, 6))

    # Create scatter plot for linear function plus noise:
    (xData, yData) = data_generator.LinearGenerator(100, noise)
    plt.scatter(xData, yData)

    # Construct iterator with linear model
    initialParameters: npFloatArray = np.random.randn(2)
    iterator = GradientDescent(
        LinearModel(initialParameters),
        xData,
        yData,
        0.6,
    )
    errorLog: list[float] = [iterator.GetCurrentCost()]

    # Start optimization:
    for i in range(5000):
        iterator.step()
        errorLog.append(iterator.GetCurrentCost())

        plt.plot(iterator.inputData, iterator.GetCurrentOutput())
        # print(iterator.GetCurrentCost())

    plt.plot(iterator.inputData, iterator.GetCurrentOutput())
    plt.figure(figsize=(8, 6))
    plt.plot(errorLog)

    print(f"Output parameters: {iterator.model.parameters}")
    print(f"Final error is {iterator.GetCurrentCost()}")
    print(f"Final derivative is {iterator.GetCurrentDeriv()}")

    plt.show()
