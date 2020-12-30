import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
from model import MLP1, MLP2
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import KFold


# Training settings
parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
					help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
					help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
					help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
					help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
					help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
					help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
					help='how many batches to wait before logging training status')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
					help='folder where experiment outputs are located.')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

# Create the folder for experiments
if not os.path.isdir(args.experiment):
	os.makedirs(args.experiment)

pretrain_model1 = 'inception_v4_iNat_448_FT_560'
pretrain_model2 = 'inception_v3_iNat_299'
pretrain_model3 = 'inception_v3_iNat_299_FT_560'
pretrain_model4 = 'inception_v4_iNat_448'
pretrain_model5 = 'inception_v3_iNat_448'

# ~~~~Model 1~~~~

load_dir1 = os.path.join('Extracted_Features', pretrain_model1)
features_train1 = np.load(os.path.join(load_dir1, pretrain_model1 + '_feature_train.npy'))
labels_train = np.load(os.path.join(load_dir1, pretrain_model1 + '_label_train.npy'))
features_val1 = np.load(os.path.join(load_dir1, pretrain_model1 + '_feature_val.npy'))
labels_val = np.load(os.path.join(load_dir1, pretrain_model1 + '_label_val.npy'))

# ~~~~Model 2~~~~

load_dir2 = os.path.join('Extracted_Features', pretrain_model2)
features_train2 = np.load(os.path.join(load_dir2, pretrain_model2 + '_feature_train.npy'))
features_val2 = np.load(os.path.join(load_dir2, pretrain_model2 + '_feature_val.npy'))

# ~~~~Model 3~~~~

load_dir3 = os.path.join('Extracted_Features', pretrain_model3)
features_train3 = np.load(os.path.join(load_dir3, pretrain_model3 + '_feature_train.npy'))
features_val3 = np.load(os.path.join(load_dir3, pretrain_model3 + '_feature_val.npy'))

# ~~~~Model 4~~~~

load_dir4 = os.path.join('Extracted_Features', pretrain_model4)
features_train4 = np.load(os.path.join(load_dir4, pretrain_model4 + '_feature_train.npy'))
features_val4 = np.load(os.path.join(load_dir4, pretrain_model4 + '_feature_val.npy'))

# ~~~~Model 5~~~~

load_dir5 = os.path.join('Extracted_Features', pretrain_model5)
features_train5 = np.load(os.path.join(load_dir5, pretrain_model5 + '_feature_train.npy'))
features_val5 = np.load(os.path.join(load_dir5, pretrain_model5 + '_feature_val.npy'))


features_all1 = np.concatenate((features_train1, features_val1))
features_all2 = np.concatenate((features_train2, features_val2))
features_all3 = np.concatenate((features_train3, features_val3))
features_all4 = np.concatenate((features_train4, features_val4))
features_all5 = np.concatenate((features_train5, features_val5))

# All extracted features are concatenated

features_all = np.concatenate((features_all1, features_all2, features_all3, features_all4, features_all5), axis=1)
labels_all = np.concatenate((labels_train, labels_val))


def train(epoch, features_train, labels_train):
	model.train()
	correct = 0.0
	data = Variable(torch.from_numpy(features_train))
	target = Variable(torch.from_numpy(labels_train)).type(torch.LongTensor)
	if use_cuda:
		data, target = data.cuda(), target.cuda()
	optimizer.zero_grad()
	output = model(data)
	criterion = torch.nn.CrossEntropyLoss(reduction='mean')
	loss = criterion(output, target)
	loss.backward()
	optimizer.step()
	pred = output.data.max(1, keepdim=True)[1]
	correct += pred.eq(target.view_as(pred)).cpu().sum()	


def validation(features_val, labels_val):
	model.eval()
	validation_loss = 0
	correct = 0
	data = Variable(torch.from_numpy(features_val))
	target = Variable(torch.from_numpy(labels_val)).type(torch.LongTensor)
	if use_cuda:
		data, target = data.cuda(), target.cuda()
	output = model(data)
	criterion = torch.nn.CrossEntropyLoss(reduction='mean')
	validation_loss += criterion(output, target).data.item()
	# get the index of the max log-probability
	pred = output.data.max(1, keepdim=True)[1]
	correct += pred.eq(target.data.view_as(pred)).cpu().sum()

	validation_loss /= features_val.shape[0]
	print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
		validation_loss, correct, features_val.shape[0],
		100. * correct / features_val.shape[0]))

	return correct 

# ~~~ 10-Fold Cross-Validation - 9/10 for the training set, 1/10 for the validation set - for measuring the accuracy on the validation set ~~~

perm = np.random.permutation(len(labels_all))

features_all = features_all[perm, :]
labels_all = labels_all[perm]
num_classes = 20

cv = KFold(n_splits=10, random_state=42, shuffle=False)

total_correct = 0

for train_index, val_index in cv.split(features_all):
	model = MLP1(features_all.shape[1], num_classes)

	if use_cuda:
		print('Using GPU')
		model.cuda()
	else:
		print('Using CPU')

	lr = args.lr
	optimizer = optim.Adam(model.parameters(), lr=lr)

	#scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
	features_train, features_val, y_train, y_val = features_all[train_index, :], features_all[val_index,:], labels_all[train_index], labels_all[val_index]

	for epoch in range(1, args.epochs + 1):

		train(epoch, features_train, y_train)

	total_correct += validation(features_val, y_val)

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Average accuracy on the validation set = ', total_correct.numpy() / len(labels_all))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model = MLP1(features_all.shape[1], num_classes)

if use_cuda:
	print('Using GPU')
	model.cuda()
else:
	print('Using CPU')

lr = args.lr
optimizer = optim.Adam(model.parameters(), lr=lr)

# scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

for epoch in range(1, args.epochs + 1):
	train(epoch, features_all, labels_all)
	model_file = args.experiment + '/model_' + str(epoch) + '.pth'
	torch.save(model.state_dict(), model_file)
