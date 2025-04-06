import numpy as np
from matplotlib import pyplot as plt

from shared.custom_typing import npFloatArray
from shared.gradient_descent import newtons_method
from shared.models import LinearModel
from shared import data_generator


def Run() -> None:
    noise = 0.2

    fig = plt.figure(figsize=(8, 6))

    # Create scatter plot for linear function plus noise:
    (xData, yData) = data_generator.LinearGenerator(3, noise)
    scatterPlot = plt.scatter(xData, yData)

    # Construct iterator with linear model
    initialParameters: npFloatArray = np.random.randn(2)
    iterator = newtons_method.NewtonOptimizer(
        LinearModel(initialParameters),
        xData,
        yData,
        1,
    )
    errorLog: list[float] = [iterator.GetCurrentCost()]

    # Start optimization:
    for i in range(20):
        iterator.step()
        errorLog.append(iterator.GetCurrentCost())

        plt.plot(iterator.inputData, iterator.GetCurrentOutput())

    fig = plt.figure(figsize=(8, 6))
    errorPlot = plt.plot(errorLog)

    print(f"Output parameters: {iterator.model.parameters}")
    plt.show()
