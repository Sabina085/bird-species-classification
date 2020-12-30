import numpy as np
import glob
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm
from data import data_transforms


'''
In this script, we construct the files train.txt, val.txt, test.txt (all in the folder 'Lists_for_feature_extraction'), which 
will be used by the feature extraction script. Train and val files contain on every row 'image_path: label' and test file contains 
on every row only the image_path.
'''


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    Source: https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]

        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


path = './bird_dataset/'
trainPath = 'train_images/'
valPath = 'val_images/'
testPath = 'test_images/mistery_category/'

# Data initialization and loading

train_loader = torch.utils.data.DataLoader(
    ImageFolderWithPaths(path + trainPath,
                         transform = data_transforms),
    batch_size=1, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(
    ImageFolderWithPaths(path + valPath,
                         transform = data_transforms),
    batch_size=1, shuffle=False, num_workers=1)

trainFile = open("Lists_for_feature_extraction/train.txt","w+")
valFile = open("Lists_for_feature_extraction/val.txt","w+")
testFile = open("Lists_for_feature_extraction/test.txt","w+")

for batch_idx, (data, target, input) in enumerate(train_loader):
    trainFile.write(input[0] + ': ' + str(target[0].numpy()) + '\n')

for batch_idx, (data, target, input) in enumerate(val_loader):
    valFile.write(input[0] + ': ' + str(target[0].numpy()) + '\n')

files = []
for r, d, f in os.walk(path + testPath):
    for file in f:
        files.append(os.path.join(r, file))

for f in files:
    testFile.write(f + '\n')
