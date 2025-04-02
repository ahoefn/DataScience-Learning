from .CustomTyping import *
import numpy as np
from numpy import typing as npt
from collections.abc import Callable


class NewtonOptimizer:
    def __init__(
        self,
        modelFunc: Callable,  # (npFloatArray,npFloatArray) -> npFloatArray
        CostFunc: Callable,  # (npFloatArray,npFloatArray) -> float
        DerivFunc: Callable,  # (npFloatArray,npFloatArray) -> npFloatArray
        parameters: npFloatArray,
        inputData: npFloatArray,
        expectedResults: npFloatArray,
        stepSize: float,
    ) -> None:
        self.modelFunc = modelFunc
        self.CostFunc = CostFunc
        self.DerivFunc = DerivFunc
        self.parameters = parameters
        self.inputData = inputData
        self.expectedResults = expectedResults
        self.stepSize = stepSize

    def GetCurrentError(self) -> float:
        return self.CostFunc(self.parameters, self.inputData, self.expectedResults)

    def step(self) -> None:
        velocityVec = -self.stepSize * self.DerivFunc(
            self.parameters, self.inputData, self.expectedResults
        )
        self.parameters += velocityVec

    def GetCurrentOutput(self) -> npFloatArray:
        return self.modelFunc(self.parameters, self.inputData)
