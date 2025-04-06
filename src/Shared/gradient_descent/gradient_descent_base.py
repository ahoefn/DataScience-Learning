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

    def GetCurrentOutputPartial(self, startIndex: int, endIndex: int) -> npFloatArray:
        return self.model.GetOutput(self.inputData[startIndex:endIndex])

    def GetCurrentCost(self) -> float:
        return self.model.GetCost(self.inputData, self.expectedResults)

    def GetCurrentCostPartial(self, startIndex: int, endIndex: int) -> float:
        partialInput: npFloatArray = self.inputData[startIndex:endIndex]
        partialResults: npFloatArray = self.expectedResults[startIndex:endIndex]
        return self.model.GetCost(partialInput, partialResults)

    def GetCurrentDeriv(self) -> npFloatArray:
        return self.model.GetDeriv(self.inputData, self.expectedResults)

    def GetCurrentDerivPartial(self, startIndex: int, endIndex: int) -> npFloatArray:
        partialInput: npFloatArray = self.inputData[startIndex:endIndex]
        partialResults: npFloatArray = self.expectedResults[startIndex:endIndex]
        return self.model.GetDeriv(partialInput, partialResults)

    def step(self) -> None:
        raise NotImplementedError
