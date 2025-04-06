from .gradient_descent_base import GradientDescentBase
from ..custom_typing import npFloatArray
from ..models import Model


class GradientDescent(GradientDescentBase):
    def __init__(
        self,
        model: Model,
        inputData: npFloatArray,
        expectedResults: npFloatArray,
        stepSize: float,
    ) -> None:
        GradientDescentBase.__init__(self, model, inputData, expectedResults)
        self.stepSize: float = stepSize

    def step(self) -> None:
        velocityVec = -self.stepSize * self.GetCurrentDeriv()
        self.model.parameters += velocityVec
