from glob import glob
from os.path import join
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

def LoadModel(name="classifier"):
    structure = Path(name+"Structure.json").read_text()
    model = keras.models.model_from_json(structure)
    model.load_weights(name+"Weights.h5")
    return model

def LoadData(batchNum=0, directory="Development"):
    fileList = glob(join("Dataset", directory, "test*"))
    dataBatch = np.load(fileList[batchNum])
    return dataBatch["data"], dataBatch["isFace"], dataBatch["labels"]

if __name__ == "__main__":
    data, isFace, labels = LoadData(directory="Training")
    data = data.astype('float32') / 255

    model = LoadModel()
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    results = model.evaluate(data, isFace) #get loss and accuracy over data

    #for current model, accuracy is 99.1% and loss is 0.0933 on test_batch_0
    #    vs the 99.81% accuracy and 0.0323 loss on train_batch_0
    