import math

from .gradient_descent_base import GradientDescentBase
from ..custom_typing import npFloatArray
from ..models.model import Model


class StochasticDescent(GradientDescentBase):
    def __init__(
        self,
        model: Model,
        inputData: npFloatArray,
        expectedResults: npFloatArray,
        stepSize: float,
        miniBatchSize: int,
    ) -> None:
        GradientDescentBase.__init__(self, model, inputData, expectedResults)
        self.stepSize: float = stepSize
        self.currentMiniBatch: int = 0
        self.miniBatchSize: int = miniBatchSize

    def step(self) -> None:
        miniBatchCount: int = math.ceil(len(self.inputData) / self.miniBatchSize) - 1
        print(miniBatchCount)
        for i in range(miniBatchCount):
            self.currentMiniBatch = i
            self.stepMiniBatch()

    # Runs a single minibatch of size self.miniBatchSize
    def stepMiniBatch(self) -> None:
        startIndex: int = self.currentMiniBatch * self.miniBatchSize
        endIndex: int = min(startIndex + self.miniBatchSize, len(self.inputData))

        velocityVec: npFloatArray = -self.stepSize * self.GetCurrentDerivPartial(
            startIndex, endIndex
        )
        self.model.parameters += velocityVec
