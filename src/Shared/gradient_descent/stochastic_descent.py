from .gradient_descent_base import GradientDescentBase
from ..custom_typing import npFloatArray
from models.model import Model


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
        self.stepSize = stepSize
        self.currentMiniBatch = 0
        self.miniBatchSize = miniBatchSize
