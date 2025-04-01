import numpy as np


class NewtonOptimizer:
    def __init__(
        this,
        CostFunc: callable,
        DerivFunc: callable,
        xData: np.ndarray[float],
        stepSize: float,
    ) -> None:
        this.CostFunc = CostFunc
        this.DerivFunc = DerivFunc
        this.currentData = xData
        this.stepSize = stepSize

    def CurrentError(this):
        return this.CostFunc(this.currentData)

    def step(this):
        velocityVec = this.stepSize * this.DerivFunc(this.currentData)
        this.currentData += velocityVec
