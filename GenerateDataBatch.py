import Faces
import Objects
import numpy as np
import argparse
from glob import glob
from os.path import join 


parser = argparse.ArgumentParser(description="Generate a mixed batch of training data")
parser.add_argument("-N", type=int, help="Number of samples to generate (assumed even split)", default=250)
parser.add_argument("-P", type=str, help="Purpose of batch (used for directory name)", default="Development")

args = parser.parse_args()
paths = []

faceCount = int(np.ceil(args.N/2))
paths.append(Faces.Pull(purpose=args.P, count=faceCount))

objCount = int(np.floor(args.N/2))
paths.append(Objects.Pull(purpose=args.P, count=objCount))

print("Merging data...")
batchNum = len(glob(join("Dataset", args.P, "batch*")))
firstPath = paths.pop(0)
currFile = np.load(firstPath+".npz")
data = currFile["data"]
isFace = currFile["isFace"]
labels = currFile["labels"]
for fileName in paths:
    currFile = np.load(fileName+".npz")
    data = np.concatenate((data, currFile["data"]), axis=0)
    isFace = np.concatenate((isFace, currFile["isFace"]), axis=0)
    labels = np.concatenate((labels, currFile["labels"]), axis=0)
mergedPath = join("Dataset", args.P, f"batch_{batchNum}-count_{args.N}")
print(f"Saving {mergedPath} to disk...")
np.savez(mergedPath,  data=data, isFace=isFace, labels=labels)
print("Done.")