import pandas as pd
from pathlib import Path


class DataHandler:
    def __init__(self) -> None:
        currentPath = Path(__file__).parent
        fileName = "SUSY.csv"
        filePath = currentPath.joinpath(fileName)
        columns = [
            "signal",
            "lepton 1 pT",
            "lepton 1 eta",
            "lepton 1 phi",
            "lepton 2 pT",
            "lepton 2 eta",
            "lepton 2 phi",
            "missing energy magnitude",
            "missing energy phi",
            "MET_rel",
            "axial MET",
            "M_R",
            "M_TR_2",
            "R",
            "MT2",
            "S_R",
            "M_Delta_R",
            "dPhi_r_b",
            "cos(theta_r1)",
        ]
        print(filePath)
        self.dataFrame = pd.read_csv(filePath, nrows=1550000, names=columns)

    def GetTrainData(self) -> pd.DataFrame:
        return self.dataFrame.iloc[:1500000]

    def GetTestData(self) -> pd.DataFrame:
        return self.dataFrame.iloc[1500000:1550000]


DataHandler()
