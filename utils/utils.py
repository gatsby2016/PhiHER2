import pickle
import torch
import numpy as np
import torch.nn as nn
import pdb

import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubsetSequentialSampler(Sampler):
	"""Samples elements sequentially from a given list of indices, without replacement.

	Arguments:
		indices (sequence): a sequence of indices
	"""
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)

def collate_MIL(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	label = torch.LongTensor(np.array([item[1] for item in batch]))
	slide_id = [item[2] for item in batch]

	return [img, label, slide_id]

def collate_features(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	coords = np.vstack([item[1] for item in batch])
	img4plip = [item[2] for item in batch]
	return [img, coords, img4plip]


def get_simple_loader(dataset, batch_size=1, num_workers=1):
	kwargs = {'num_workers': 4, 'pin_memory': False, 'num_workers': num_workers} if device.type == "cuda" else {}
	loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_MIL, **kwargs)
	return loader 

def get_split_loader(split_dataset, training = False, testing = False, weighted = False, surv=False, batch_size=1):
	"""
		return either the validation loader or training loader 
	"""
	if surv:
		collate = collate_MIL_survival
	else:
		collate = collate_MIL

	kwargs = {'num_workers': 4} if device.type == "cuda" else {}
	if not testing:
		if training:
			if weighted:
				weights = make_weights_for_balanced_classes_split(split_dataset)
				loader = DataLoader(split_dataset, batch_size=batch_size, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate, **kwargs)	
			else:
				loader = DataLoader(split_dataset, batch_size=batch_size, sampler = RandomSampler(split_dataset), collate_fn = collate, **kwargs)
		else:
			loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SequentialSampler(split_dataset), collate_fn = collate, **kwargs)

	else:
		ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
		loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SubsetSequentialSampler(ids), collate_fn = collate, **kwargs )

	return loader

def get_optim(model, args):
	if args.opt == "adam":
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
	elif args.opt == 'sgd':
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
	else:
		raise NotImplementedError
	return optimizer

def print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, val_num, test_num, samples, n_splits = 5, label_frac = 1.0, custom_test_ids = None):
	indices = np.arange(samples).astype(int)
	
	if custom_test_ids is not None:
		indices = np.setdiff1d(indices, custom_test_ids)

	for i in range(n_splits):
		all_val_ids = []
		all_test_ids = []
		sampled_train_ids = []
		
		if custom_test_ids is not None: # pre-built test split, do not need to sample
			all_test_ids.extend(custom_test_ids)

		for c in range(len(val_num)):
			possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class
			val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids

			remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
			all_val_ids.extend(val_ids)

			if custom_test_ids is None: # sample test split

				test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
				remaining_ids = np.setdiff1d(remaining_ids, test_ids)
				all_test_ids.extend(test_ids)

			if label_frac == 1:
				sampled_train_ids.extend(remaining_ids)
			
			else:
				sample_num  = math.ceil(len(remaining_ids) * label_frac)
				slice_ids = np.arange(sample_num)
				sampled_train_ids.extend(remaining_ids[slice_ids])

		yield sampled_train_ids, all_val_ids, all_test_ids

"""
generate_split_CrossValidation() 
与上述代码的区别在于，上述每一split的val存在重叠交叉；而标准的交叉验证，是先分n folds，每一fold不overlap，然后每一fold作为val；下述实现
"""
def generate_split_CrossValidation(cls_ids, test_num, samples, n_splits = 10, custom_test_ids = None): 

    indices = np.arange(samples).astype(int)

    if custom_test_ids is not None:
        all_test_ids = custom_test_ids # pre-built test split, do not need to sample
        indices = np.setdiff1d(indices, custom_test_ids) #indices 移除特定的test ids
    else: # custom_test_ids is None
        all_test_ids = []
        for c in range(len(test_num)):
            test_ids = np.random.choice(cls_ids[c], test_num[c], replace = False)
            all_test_ids.extend(test_ids)

        indices = np.setdiff1d(indices, all_test_ids) # 先把all_test_ids都确定下来，然后剩下的进行kfold

    new_cls_ids = []
    for c in range(len(test_num)): # 所有类别的样本取交集后打乱；然后取 0~N/k, N/k~N/K*2,
        new_cls_ids.append(np.intersect1d(cls_ids[c], indices)) #all indices of this class
        np.random.shuffle(new_cls_ids[c])

    start_idx = np.zeros(len(new_cls_ids), dtype=np.int32)
    for _ in range(n_splits):
        all_val_ids = []
        sampled_train_ids = []
        
        for c in range(len(test_num)):
            val = len(new_cls_ids[c]) / n_splits # 计算step小数
            step = np.random.choice([math.floor(val), math.ceil(val), round(val)], 1)[0] # 小数取上取下 随机，作为step
            
            val_ids = new_cls_ids[c][start_idx[c]: min(start_idx[c] + step, len(new_cls_ids[c]))] # 取id
            all_val_ids.extend(val_ids)
            remaining_ids = np.setdiff1d(new_cls_ids[c], all_val_ids) #indices of this class left: Return the unique values in `ar1` that are not in `ar2`.
            sampled_train_ids.extend(remaining_ids)
            
            start_idx[c] += step

        yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
	if n is None:
		return collections.deque(iterator, maxlen=0)
	else:
		return next(islice(iterator,n, None), default)

def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

	return error

def make_weights_for_balanced_classes_split(dataset):
	N = float(len(dataset))                                           
	weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
	weight = [0] * int(N)                                           
	for idx in range(len(dataset)):   
		y = dataset.getlabel(idx)                        
		weight[idx] = weight_per_class[y]                                  

	return torch.DoubleTensor(weight)


# ad from Porpoise project
def collate_pathol_factor_surv(batch):
    img = torch.cat([item[0] for item in batch], dim=0).reshape(len(batch), -1)
    slideids = [item[1] for item in batch]
    label = torch.Tensor([item[2] for item in batch]).type(torch.LongTensor)
    event_time = torch.FloatTensor([item[3] for item in batch])
    c = torch.FloatTensor([item[4] for item in batch])
    return [img, slideids, label, event_time, c]


def collate_MIL_survival(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    omic = torch.cat([item[1] for item in batch], dim = 0).type(torch.FloatTensor)
    label = torch.Tensor([item[2] for item in batch]).type(torch.LongTensor)
    event_time = torch.FloatTensor([item[3] for item in batch])
    c = torch.FloatTensor([item[4] for item in batch])
    return [img, omic, label, event_time, c]

def collate_MIL_survival_cluster(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    cluster_ids = torch.cat([item[1] for item in batch], dim = 0).type(torch.LongTensor)
    omic = torch.cat([item[2] for item in batch], dim = 0).type(torch.FloatTensor)
    label = torch.LongTensor([item[3] for item in batch])
    event_time = np.array([item[4] for item in batch])
    c = torch.FloatTensor([item[5] for item in batch])
    return [img, cluster_ids, omic, label, event_time, c]

def collate_MIL_survival_sig(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    omic1 = torch.cat([item[1] for item in batch], dim = 0).type(torch.FloatTensor)
    omic2 = torch.cat([item[2] for item in batch], dim = 0).type(torch.FloatTensor)
    omic3 = torch.cat([item[3] for item in batch], dim = 0).type(torch.FloatTensor)
    omic4 = torch.cat([item[4] for item in batch], dim = 0).type(torch.FloatTensor)
    omic5 = torch.cat([item[5] for item in batch], dim = 0).type(torch.FloatTensor)
    omic6 = torch.cat([item[6] for item in batch], dim = 0).type(torch.FloatTensor)

    label = torch.LongTensor([item[7] for item in batch])
    event_time = np.array([item[8] for item in batch])
    c = torch.FloatTensor([item[9] for item in batch])
    return [img, omic1, omic2, omic3, omic4, omic5, omic6, label, event_time, c]


def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


def dfs_unfreeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
        dfs_unfreeze(child)


def l1_reg_all(model, reg_type=None):
    l1_reg = None

    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)
    return l1_reg

def l1_reg_modules(model, reg_type=None):
    l1_reg = 0

    l1_reg += l1_reg_all(model.fc_omic)
    l1_reg += l1_reg_all(model.mm)

    return l1_reg

def l1_reg_omic(model, reg_type=None):
    l1_reg = 0

    if hasattr(model, 'fc_omic'):
        l1_reg += l1_reg_all(model.fc_omic)
    else:
        l1_reg += l1_reg_all(model)

    return l1_reg

def get_custom_exp_code(args):
    r"""
    Updates the argparse.NameSpace with a custom experiment code.

    Args:
        - args (NameSpace)

    Returns:
        - args (NameSpace)
    """
    exp_code = '_'.join(args.split_dir.split('_')[:2])
    dataset_path = 'datasets_csv'
    param_code = ''

    ### Model Type
    if args.model_type == 'porpoise_mmf':
      param_code += 'PorpoiseMMF'
    elif args.model_type == 'porpoise_amil':
      param_code += 'PorpoiseAMIL'
    elif args.model_type == 'max_net' or args.model_type == 'snn':
      param_code += 'SNN'
    elif args.model_type == 'amil':
      param_code += 'AMIL'
    elif args.model_type == 'deepset':
      param_code += 'DS'
    elif args.model_type == 'mi_fcn':
      param_code += 'MIFCN'
    elif args.model_type == 'mcat':
      param_code += 'MCAT'
    else:
      raise NotImplementedError

    ### Loss Function
    param_code += '_%s' % args.bag_loss
    if args.bag_loss in ['nll_surv']:
        param_code += '_a%s' % str(args.alpha_surv)

    ### Learning Rate
    if args.lr != 2e-4:
      param_code += '_lr%s' % format(args.lr, '.0e')

    ### L1-Regularization
    if args.reg_type != 'None':
      param_code += '_%sreg%s' % (args.reg_type, format(args.lambda_reg, '.0e'))

    if args.dropinput:
      param_code += '_drop%s' % str(int(args.dropinput*100))

    param_code += '_%s' % args.which_splits.split("_")[0]

    ### Batch Size
    if args.batch_size != 1:
      param_code += '_b%s' % str(args.batch_size)

    ### Gradient Accumulation
    if args.gc != 1:
      param_code += '_gc%s' % str(args.gc)

    ### Applying Which Features
    if args.apply_sigfeats:
      param_code += '_sig'
      dataset_path += '_sig'
    elif args.apply_mutsig:
      param_code += '_mutsig'
      dataset_path += '_mutsig'

    ### Fusion Operation
    if args.fusion != "None":
      param_code += '_' + args.fusion

    ### Updating
    args.exp_code = exp_code + "_" + param_code
    args.param_code = param_code
    args.dataset_path = dataset_path

    return args