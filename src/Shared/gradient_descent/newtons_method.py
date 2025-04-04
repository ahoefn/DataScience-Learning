from .gradient_descent_base import GradientDescentBase
from ..custom_typing import npFloatArray
from collections.abc import Callable


class NewtonOptimizer(GradientDescentBase):
    def __init__(
        self,
        ModelFunc: Callable,  # (npFloatArray,npFloatArray) -> npFloatArray
        CostFunc: Callable,  # (npFloatArray,npFloatArray) -> float
        DerivFunc: Callable,  # (npFloatArray,npFloatArray) -> npFloatArray
        parameters: npFloatArray,
        inputData: npFloatArray,
        expectedResults: npFloatArray,
        stepSize: float,
    ) -> None:
        GradientDescentBase.__init__(
            self, ModelFunc, CostFunc, DerivFunc, parameters, inputData, expectedResults
        )
        self.stepSize = stepSize

    def step(self) -> None:
        velocityVec = -self.stepSize * self.DerivFunc(
            self.parameters, self.inputData, self.expectedResults
        )
        self.parameters += velocityVec
