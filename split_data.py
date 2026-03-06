import os
import random
import shutil
from itertools import islice

outputFolderPath = "Dataset/SplitData"
inputFolderPath = "Dataset/all"
splitRatio = {"train": 0.7, "val": 0.2, "test": 0.1}
classes = ["fake", "real"]

# Remove old split folder
if os.path.exists(outputFolderPath):
    shutil.rmtree(outputFolderPath)

# Create directories
for split in splitRatio.keys():
    os.makedirs(f"{outputFolderPath}/{split}/images", exist_ok=True)
    os.makedirs(f"{outputFolderPath}/{split}/labels", exist_ok=True)

# Get unique names
listNames = os.listdir(inputFolderPath)
uniqueNames = list(set([name.split('.')[0] for name in listNames]))

random.shuffle(uniqueNames)

lenData = len(uniqueNames)
lenTrain = int(lenData * splitRatio['train'])
lenVal = int(lenData * splitRatio['val'])
lenTest = lenData - lenTrain - lenVal

lengthToSplit = [lenTrain, lenVal, lenTest]
Input = iter(uniqueNames)
Output = [list(islice(Input, elem)) for elem in lengthToSplit]

print(f"Total Images: {lenData}")
print(f"Train: {lenTrain}, Val: {lenVal}, Test: {lenTest}")

sequence = ['train', 'val', 'test']

for i, out in enumerate(Output):
    for fileName in out:
        shutil.copy(
            f'{inputFolderPath}/{fileName}.jpg',
            f'{outputFolderPath}/{sequence[i]}/images/{fileName}.jpg'
        )
        shutil.copy(
            f'{inputFolderPath}/{fileName}.txt',
            f'{outputFolderPath}/{sequence[i]}/labels/{fileName}.txt'
        )

# Create data.yaml
dataYaml = f"""
path: ../SplitData
train: train/images
val: val/images
test: test/images

nc: {len(classes)}
names: {classes}
"""

with open(f"{outputFolderPath}/data.yaml", 'w') as f:
    f.write(dataYaml)

print("Split Completed & data.yaml Created")