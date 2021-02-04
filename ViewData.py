import numpy as np
import matplotlib.pyplot as plt
from sys import argv

pathName = argv[1]
batch = np.load(pathName)

def showImage(index = 0):
    plt.imshow(batch["data"][index])
    plt.title(f"Image: {index}, isFace: {batch['isFace'][index]}, label: {batch['labels'][index]}")
    plt.show()

showImage()