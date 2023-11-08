from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle

from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils, models
import torch.nn.functional as F

from PIL import Image
import h5py

from random import randrange

def eval_transforms(pretrained=False, gaussian_blur=False, set_size=None):
    mean = (0.5,0.5,0.5)
    std = (0.5,0.5,0.5)
	
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    if gaussian_blur:
        trnsfrms_val = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.GaussianBlur(kernel_size=3, sigma=1),    
                    transforms.Normalize(mean = mean, std = std)
                ]
            )
    else:
        trnsfrms_val = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(mean = mean, std = std)
                    ]
                )	
    if set_size is not None:
        trnsfrms_val = transforms.Compose([transforms.Resize(size=set_size)] + trnsfrms_val.transforms)
		
    return trnsfrms_val


class Whole_Slide_Bag(Dataset):
	def __init__(self,
		file_path,
		pretrained=False,
		gaussian_blur=False,
		resize_size=None,
		custom_transforms=None,
		target_patch_size=-1,
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.pretrained=pretrained
		if target_patch_size > 0:
			self.target_patch_size = (target_patch_size, target_patch_size)
		else:
			self.target_patch_size = None

		if not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained, gaussian_blur=gaussian_blur,
										set_size=resize_size)
		else:
			self.roi_transforms = custom_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['imgs']
			self.length = len(dset)

		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['imgs']
		for name, value in dset.attrs.items():
			print(name, value)

		print('pretrained:', self.pretrained)
		print('transformations:', self.roi_transforms)
		if self.target_patch_size is not None:
			print('target_size: ', self.target_patch_size)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			img = hdf5_file['imgs'][idx]
			coord = hdf5_file['coords'][idx]
		
		img = Image.fromarray(img)
		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
		img = self.roi_transforms(img).unsqueeze(0)
		return img, coord

class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
		file_path,
		wsi,
		pretrained=False,
		gaussian_blur=False,
		resize_size=None,
		custom_transforms=None,
		custom_downsample=1,
		target_patch_size=-1
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
			custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
			target_patch_size (int): Custom defined image size before embedding
		"""
		self.pretrained=pretrained
		self.wsi = wsi
		self.plip_trans = transforms.Compose([transforms.Resize(size=224)])
		if not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained, gaussian_blur=gaussian_blur,
										 set_size=resize_size)
		else:
			self.roi_transforms = custom_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
			if target_patch_size > 0:
				self.target_patch_size = (target_patch_size, ) * 2
			elif custom_downsample > 1:
				self.target_patch_size = (self.patch_size // custom_downsample, ) * 2
			else:
				self.target_patch_size = None
		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		self.dset = hdf5_file['coords']
		for name, value in self.dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('target patch size: ', self.target_patch_size) if self.target_patch_size is not None else print('target patch size: ',(self.patch_size, )*2 )
		print('pretrained: ', self.pretrained)
		print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx): 
		# with h5py.File(self.file_path,'r') as hdf5_file:
			# coord = hdf5_file['coords'][idx]
		coord = self.dset[idx] # 修改频繁读写h5 file为读取一次，保留到类中
		
		# FIXME 1410114HE.mrxs 样本的11543 idx，首次遇到 openslide read_region 报错'openslide.lowlevel.OpenSlideError Not a JPEG file: starts with 0xc3 0xcf'
		# 如issue陈述： https://github.com/openslide/openslide/pull/211
		# 确定是openslide和数据的问题，目前不对openslide源码修改，仅在这里做判断，然后create new (0, 0, 0) image代替，并修改coord为[-1, -1]进行标记; 不影响整个batch
		try:
			img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
		except:
			print("openslide.lowlevel.OpenSlideError Not a JPEG file: starts with 0xc3 0xcf', Now idx: ", idx)
			img = Image.new(size=(self.patch_size, self.patch_size), mode="RGB", color=(0,0,0))
			coord = np.array([-1, -1]) # 修改coord 坐标为（-1,-1）进行标记

		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
		image = self.roi_transforms(img).unsqueeze(0)
		
		img4plip = self.plip_trans(img)

		return image, coord, img4plip

class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]




