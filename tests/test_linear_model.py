import unittest
import numpy as np

from src.chapters.shared.models.mean_squared_error import MeanSqrdError
from src.chapters.shared.models.linear_model import MeanSqrdErrorDerivative
from src.chapters.shared.custom_typing import npFloatArray
from src.chapters.shared.models import LinearModel


class TestLinearModel(unittest.TestCase):
    def test_AtOptimum(self) -> None:
        # Setup:
        xData: npFloatArray = np.array([0, 1])
        yData: npFloatArray = np.array([1, 3])
        parameters: npFloatArray = np.array([1, 2])
        model = LinearModel(parameters)

        # Start tests:
        np.testing.assert_almost_equal(model.GetOutput(xData), yData)
        np.testing.assert_almost_equal(model.GetCost(xData, yData), 0)

        np.testing.assert_almost_equal(
            MeanSqrdErrorDerivative(parameters, xData, yData), np.array([0, 0])
        )
        np.testing.assert_almost_equal(model.GetDeriv(xData, yData), np.array([0, 0]))

    def test_NotAtOptimum(self) -> None:
        # Setup:
        xData: npFloatArray = np.array([0, 1, 2])
        yData: npFloatArray = np.array([-1, 4, 9])
        parameters: npFloatArray = np.array([0, 1])
        model = LinearModel(parameters)

        # Start tests:
        expectedOutput: npFloatArray = np.array([0, 1, 2])
        np.testing.assert_almost_equal(model.GetOutput(xData), expectedOutput)

        expectedCost = ((0 - 1) ** 2 + (1 - 4) ** 2 + (2 - 9) ** 2) / 3
        np.testing.assert_almost_equal(
            expectedCost, MeanSqrdError(expectedOutput, yData)
        )
        np.testing.assert_almost_equal(model.GetCost(xData, yData), expectedCost)

        # Calculated manually
        expectedDeriv: npFloatArray = np.array([-6, -34 / 3])

        np.testing.assert_almost_equal(
            MeanSqrdErrorDerivative(parameters, xData, yData), expectedDeriv
        )
        np.testing.assert_almost_equal(model.GetDeriv(xData, yData), expectedDeriv)


if __name__ == "__main__":
    unittest.main()
