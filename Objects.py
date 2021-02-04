import numpy as np
from glob import glob
from os.path import join

def Pull(purpose="Development", count=250):
    #purpose: pull data for Training, Evaluation, etc
    #count: number of samples to pull
    batchNum = len(glob(join("Dataset", purpose, "objects-*")))

    isFace = [] #Binary classification: 1 contains face; 0 no face
    labels = [] #Empty list of data labels from original dataset

    print(f"Pulling {count} objects...")

    imageNet = np.load("./ImageNet/train_data_batch_1.npz", mmap_mode="r") #ImageNet 64x64 Train part 1 batch 1
    data = imageNet["data"][0:count].reshape((-1,64,64,3), order = 'F')
    isFace = np.zeros(count).astype("uint8")
    labels = imageNet["labels"][:count]
    
    #save in numpy zipped format with labels on each array
    dataPath = join("Dataset", purpose, f"objects-batch_{batchNum}-count_{count}")
    print(f"Saving {dataPath} to disk...")
    np.savez(dataPath, data=data, isFace=isFace, labels=labels)
    print("Saved.")
    del isFace, labels, data
    return dataPath

if __name__ == "__main__":
    Pull()