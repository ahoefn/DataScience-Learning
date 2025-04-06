from ..custom_typing import npFloatArray
from ..models.model import Model


class GradientDescentBase:
    def __init__(
        self,
        model: Model,
        inputData: npFloatArray,
        expectedResults: npFloatArray,
    ) -> None:
        self.model: Model = model
        self.inputData: npFloatArray = inputData
        self.expectedResults: npFloatArray = expectedResults

    def GetCurrentOutput(self) -> npFloatArray:
        return self.model.GetOutput(self.inputData)

    def GetCurrentCost(self) -> float:
        return self.model.GetCost(self.inputData, self.expectedResults)

    def GetCurrentDeriv(self) -> npFloatArray:
        return self.model.GetDeriv(self.inputData, self.expectedResults)

    def step(self) -> None:
        raise NotImplementedError
