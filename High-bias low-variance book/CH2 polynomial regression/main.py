import numpy as np
import DataGenerator
from matplotlib import pyplot as plt

fig = plt.figure(figsize=(8, 6))
data = DataGenerator.LinearGenerator(10, 0.1)
print(data)
p1 = plt.scatter(data[0], data[1])
