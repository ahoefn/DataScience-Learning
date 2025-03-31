import numpy as np


def LinearGenerator(
    sampleCount: int, standardDev: float
) -> tuple[np.ndarray, np.ndarray]:
    xData = np.linspace(0.05, 0.95, sampleCount)
    variances = standardDev * np.random.randn(sampleCount)
    yData = 2 * xData + variances
    return (xData, yData)
