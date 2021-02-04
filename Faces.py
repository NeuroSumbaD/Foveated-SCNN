import numpy as np
from glob import glob
from PIL import Image
from os.path import split, join

def Pull(purpose="Development", count=250):
    #purpose: pull data for Training, Evaluation, etc
    #count: number of samples to pull
    batchNum = len(glob(join("Dataset", purpose, "faces-*")))

    isFace = [] #Binary classification: 1 contains face; 0 no face
    labels = [] #Empty list of data labels

    print(f"Pulling {count} faces...")

    LFW = glob( join("LFW","lfw","*") ) # Labelled Faces in the Wild list of included names
    trueIndices = np.random.randint(0, len(LFW), count) 
    trueSamples = [] #samples which DO include faces
    for index in trueIndices:
        subjectPath = LFW[index] # path to a single subject in the dataset
        subjectName = split(subjectPath)[1]
        labels.append(subjectName) 
        isFace.append(1)
        imgPath = glob(join(subjectPath, "*"))[0] #path of single image belonging to subject
        img = Image.open(imgPath)
        downsample = img.resize((64,64), resample=Image.BOX)
        # Image.BOX is used to match downsampling method in ImageNet library
        npForm = np.array(downsample.getdata()).reshape((64,64,3)) #convert to correct shape and type
        trueSamples.append(npForm)

    print("Converting to numpy format...")
    data = np.array(trueSamples) # array of extended lists
    del trueSamples, LFW

    # numpy array conversion
    isFace = np.array(isFace)
    labels = np.array(labels)

    #save in numpy zipped format with labels on each array
    dataPath = join("Dataset", purpose, f"faces-batch_{batchNum}-count_{count}")
    print(f"Saving {dataPath} to disk...")
    np.savez(dataPath, data=data, isFace=isFace, labels=labels)
    print("Saved.")
    del isFace, labels, data
    return dataPath

if __name__ == "__main__":
    Pull()