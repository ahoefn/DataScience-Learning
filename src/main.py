import sys

from chapter2_linear_regression import ch2_linear_regression
from chapter4_gradient_descent import ch4_newtons_method


def RunChapter(chapter: str):
    match chapter:
        case "-ch2":
            ch2_linear_regression.Run()

        case "-ch4-newt":
            ch4_newtons_method.Run()

        case _:
            print("Invalid argument, valid arguments are: ch2, ch4-newt")
            raise ValueError


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Please select a chapter to run by adding a suitable argument.")
        print("Valid chapter arguments are:")
        print("'-ch2' running a linear regression module")
        print("'-ch4-newt' running an optimizer using newtons method")
    for arg in sys.argv[1:]:
        RunChapter(arg)
