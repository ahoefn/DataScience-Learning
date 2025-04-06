import unittest
import numpy as np

from src.shared.gradient_descent.gradient_descent import NewtonOptimizer
from src.shared.models import LinearModel
from src.shared.custom_typing import npFloatArray


class TestNewtonOptimizer(unittest.TestCase):
    def setUp(self) -> None:
        xData: npFloatArray = np.array([0, 1, 2])
        yData: npFloatArray = np.array([-1, 4, 9])
        parameters: npFloatArray = np.array([-1, 3])
        self.newton = NewtonOptimizer(LinearModel(parameters), xData, yData, 1 / 2)

    def test_GetCurrentOutput(self) -> None:
        expectedOutput: npFloatArray = np.array([-1, 2, 5])
        np.testing.assert_almost_equal(self.newton.GetCurrentOutput(), expectedOutput)

    def test_GetCurrentCost(self) -> None:
        expectedCost: float = 20 / 3
        np.testing.assert_almost_equal(self.newton.GetCurrentCost(), expectedCost)

    def test_GetCurrentDeriv(self) -> None:
        expectedDeriv: npFloatArray = np.array([-4, -20 / 3])
        np.testing.assert_almost_equal(self.newton.GetCurrentDeriv(), expectedDeriv)

    def test_Step(self) -> None:
        expectedParams: npFloatArray = np.array([3, 3 + 10 / 3])
        self.newton.step()
        outputParams: npFloatArray = self.newton.model.parameters
        np.testing.assert_almost_equal(outputParams, expectedParams)


if __name__ == "__main__":
    unittest.main()
