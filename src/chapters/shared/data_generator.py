from .custom_typing import npFloatArray
import numpy as np


def LinearGenerator(
    sampleCount: int, standardDev: float
) -> tuple[npFloatArray, npFloatArray]:
    xData: npFloatArray = np.linspace(0.05, 0.95, sampleCount, dtype=float)
    variances = standardDev * np.random.randn(sampleCount)
    yData = 2 * xData + variances
    return (xData, yData)
