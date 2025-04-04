from ..custom_typing import npFloatArray
from collections.abc import Callable


class GradientDescentBase:
    def __init__(
        self,
        ModelFunc: Callable,  # (npFloatArray,npFloatArray) -> npFloatArray
        CostFunc: Callable,  # (npFloatArray,npFloatArray) -> float
        DerivFunc: Callable,  # (npFloatArray,npFloatArray) -> npFloatArray
        parameters: npFloatArray,
        inputData: npFloatArray,
        expectedResults: npFloatArray,
    ) -> None:
        self.ModelFunc = ModelFunc
        self.CostFunc = CostFunc
        self.DerivFunc = DerivFunc
        self.parameters = parameters
        self.inputData = inputData
        self.expectedResults = expectedResults

    def GetCurrentError(self) -> float:
        return self.CostFunc(self.parameters, self.inputData, self.expectedResults)

    def step(self) -> None:
        raise NotImplementedError

    def GetCurrentOutput(self) -> npFloatArray:
        return self.ModelFunc(self.parameters, self.inputData)
