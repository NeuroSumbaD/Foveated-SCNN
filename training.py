import numpy as np
import tensorflow as tf
from glob import glob
from os.path import join

def CreateLayers(numConv=3, activation='relu', finalLayer=None):
    layers = []
    layers.append(tf.keras.Input(shape=(64,64,3)))
    for index in range(numConv):
        layers.append(
            tf.keras.layers.Conv2D(filters=16*0.5**index, kernel_size=(3,3), use_bias=False,
            strides=(2,2), padding="valid", activation=activation)(layers[-1])
        )
    if finalLayer != None:
        layers.append(tf.keras.layers.Flatten()(layers[-1]))
        layers.append(finalLayer(layers[-1]))
    return layers

def CreateModel(layers):
    return tf.keras.Model(inputs=layers[0], outputs=layers[-1])

def LoadData(batchNum=0, directory="Development"):
    fileList = glob(join("Dataset", directory, "train*"))
    dataBatch = np.load(fileList[batchNum])
    return dataBatch["data"], dataBatch["isFace"], dataBatch["labels"]


if __name__ == "__main__":
    data, isFace, labels = LoadData(directory="Training")
    data = data.astype('float32') / 255

    model = CreateModel(CreateLayers(finalLayer=tf.keras.layers.Dense(1, activation="relu")))
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(data, isFace, epochs=100)
    with open("classifierStructure.json", "w") as f:
        f.write(model.to_json())
    model.save_weights("classifierWeights.h5")

    import matplotlib.pyplot as plt
    plt.plot(history.history["loss"])
    plt.title("Model Loss")
    plt.xlabel("epochs")
    plt.ylabel("Binary Crossentropy")
    plt.savefig("Model Loss.png")
    plt.show()

    plt.plot(history.history["accuracy"])
    plt.title("Model accuracy")
    plt.xlabel("epochs")
    plt.ylabel("Percent correct")
    plt.savefig("Model Accuracy.png")
    plt.show()