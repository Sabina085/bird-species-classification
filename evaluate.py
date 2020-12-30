import argparse
from tqdm import tqdm
import os
import PIL.Image as Image
import numpy as np 
import torch
from torch.autograd import Variable
from model import  MLP1, MLP2


parser = argparse.ArgumentParser(description='RecVis A3 evaluation script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")
parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='experiment/kaggle.csv', metavar='D',
                    help="name of the output csv file")

args = parser.parse_args()
use_cuda = torch.cuda.is_available()
state_dict = torch.load(args.model)

pretrain_model1 = 'inception_v4_iNat_448_FT_560'
pretrain_model2 = 'inception_v3_iNat_299'
pretrain_model3 = 'inception_v3_iNat_299_FT_560'
pretrain_model4 = 'inception_v4_iNat_448'
pretrain_model5 = 'inception_v3_iNat_448'

load_dir1 = os.path.join('Extracted_Features', pretrain_model1)
load_dir2 = os.path.join('Extracted_Features', pretrain_model2)
load_dir3 = os.path.join('Extracted_Features', pretrain_model3)
load_dir4 = os.path.join('Extracted_Features', pretrain_model4)
load_dir5 = os.path.join('Extracted_Features', pretrain_model5)

features_test1 = np.load(os.path.join(load_dir1, pretrain_model1 + '_feature_test.npy'))
features_test2 = np.load(os.path.join(load_dir2, pretrain_model2 + '_feature_test.npy'))
features_test3 = np.load(os.path.join(load_dir3, pretrain_model3 + '_feature_test.npy'))
features_test4 = np.load(os.path.join(load_dir4, pretrain_model4 + '_feature_test.npy'))
features_test5 = np.load(os.path.join(load_dir5, pretrain_model5 + '_feature_test.npy'))

features_test = np.concatenate((features_test1, features_test2, features_test3, features_test4, features_test5), axis=1)
features_test = Variable(torch.from_numpy(features_test))

model = MLP1(features_test.shape[1], 20)

model.load_state_dict(state_dict)
model.eval()

output_file = open(args.outfile, "w")
output_file.write("Id,Category\n")

output = model(features_test)
pred = output.data.max(1, keepdim=True)[1]

'''
In the file test.txt, we saved all the test image paths
and, from these paths, we extracted the names of the test images.
This file was also used for the feature extraction task (feature_extraction script).
'''

i = 0

for f in open('Lists_for_feature_extraction/test.txt', 'r'):
    f = f.strip().split("/")[-1]
    output_file.write("%s,%d\n" % (f[:-4], pred[i]))
    i += 1

output_file.close()

print("Succesfully wrote " + args.outfile + ', you can upload this file to the kaggle competition website')
        