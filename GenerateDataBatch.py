import Faces
import Objects
import numpy as np
import argparse
from glob import glob
from os.path import join 


parser = argparse.ArgumentParser(description="Generate a mixed batch of training data")
parser.add_argument("-N", type=int, help="Number of samples to generate (assumed even split)", default=256)
parser.add_argument("-T", help="Prepare training and test data", default="False", action="store_true")
parser.add_argument("--split", type=float, help="training to test split (0 to 1)", default=0.8)

args = parser.parse_args()
purpose = "Training" if args.T else "Development"
paths = []

faceCount = int(np.ceil(args.N/2))
paths.append(Faces.Pull(purpose=purpose, count=faceCount))

objCount = int(np.floor(args.N/2))
paths.append(Objects.Pull(purpose=purpose, count=objCount))

print("Merging data...")
batchNum = len(glob(join("Dataset", purpose, "batch*")))
#Initialize numpy arrays
firstPath = paths.pop(0)
currFile = np.load(firstPath+".npz")
count = int(np.ceil(len(currFile["data"])*args.split)) if args.T else args.N
data, isFace, labels = currFile["data"][:count], currFile["isFace"][:count], currFile["labels"][:count]
testData, testIsFace, testLabels = currFile["data"][count:], currFile["isFace"][count:], currFile["labels"][count:]
#Merge arrays
for fileName in paths:
    currFile = np.load(fileName+".npz")
    count = int(np.ceil(len(currFile["data"])*args.split)) if args.T else args.N
    data = np.concatenate((data, currFile["data"][:count]), axis=0)
    isFace = np.concatenate((isFace, currFile["isFace"][:count]), axis=0)
    labels = np.concatenate((labels, currFile["labels"][:count]), axis=0)
    
    if args.T: #skip if no training/test split
        testData = np.concatenate((testData, currFile["data"][count:]), axis=0)
        testIsFace = np.concatenate((testIsFace, currFile["isFace"][count:]), axis=0)
        testLabels = np.concatenate((testLabels, currFile["labels"][count:]), axis=0)

mergedPath = join("Dataset", purpose, f"train_batch_{batchNum}-count_{len(data)}")
print(f"Saving {mergedPath} to disk...")
np.savez(mergedPath,  data=data, isFace=isFace, labels=labels)

if args.T: #Skipt if no training/test split
    testPath = join("Dataset", purpose, f"test_batch_{batchNum}-count_{len(testData)}")
    print(f"Saving {testPath} to disk...")
    np.savez(testPath,  data=testData, isFace=testIsFace, labels=testLabels)

print("Done.")