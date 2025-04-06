import sys

from chapter2_linear_regression import ch2_linear_regression
from chapter4_gradient_descent import ch4_gradient_descent


def RunChapter(chapter: str):
    match chapter:
        case "-ch2":
            ch2_linear_regression.Run()

        case "-ch4-grad":
            ch4_gradient_descent.Run()

        case _:
            print("Invalid argument, valid arguments are: ch2, ch4-grad")
            raise ValueError


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Please select a chapter to run by adding a suitable argument.")
        print("Valid chapter arguments are:")
        print("'-ch2' running a linear regression module")
        print("'-ch4-grad' optimizer using gradient descent ")
    for arg in sys.argv[1:]:
        RunChapter(arg)
