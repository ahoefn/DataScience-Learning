from ..custom_typing import npFloatArray
from collections.abc import Callable


# A class combining a model function with the error and derivative functions
class Model:
    def __init__(
        self,
        # (parameters: npFloatArray,data: npFloatArray) -> npFloatArray
        ModelFunc: Callable[[npFloatArray, npFloatArray], npFloatArray],
        # (inputData: npFloatArray,expectedOutput: npFloatArray) -> float
        CostFunc: Callable[[npFloatArray, npFloatArray], float],
        # (inputData: npFloatArray,expectedOutput: npFloatArray) -> npFloatArray
        DerivFunc: Callable[[npFloatArray, npFloatArray, npFloatArray], npFloatArray],
        parameters: npFloatArray,
    ) -> None:
        self.ModelFunc = ModelFunc
        self.CostFunc = CostFunc
        self.DerivFunc = DerivFunc
        self.parameters: npFloatArray = parameters

    def GetOutput(self, inputData: npFloatArray) -> npFloatArray:
        return self.ModelFunc(self.parameters, inputData)

    def GetCost(self, inputData: npFloatArray, expectedOutput: npFloatArray) -> float:
        currentOutput: npFloatArray = self.GetOutput(inputData)
        return self.CostFunc(currentOutput, expectedOutput)

    def GetDeriv(
        self, inputData: npFloatArray, expectedOutput: npFloatArray
    ) -> npFloatArray:
        return self.DerivFunc(self.parameters, inputData, expectedOutput)
