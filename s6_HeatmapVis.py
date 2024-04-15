from __future__ import print_function

import numpy as np

import argparse

import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
from utils.eval_utils import initiate_model as initiate_model
from models.model_CLAM import CLAM_MB, CLAM_SB
from models.model_ABMIL import ABMIL
from models.model_ProtoTrans import PhiHER2model


from types import SimpleNamespace
from collections import namedtuple
import h5py
import yaml
from wsi_core.batch_process_utils import initialize_df
from utils.heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches
from wsi_core.wsi_utils import sample_rois
from utils.file_utils import save_hdf5
import utils.config_utils as cfg_utils


def get_parser():
    parser = argparse.ArgumentParser(description='Heatmap inference script')
    parser.add_argument('--config_file', type=str, default="scripts/HE2HER2/cfgs/heatmap_cfgs_template.yaml")
    parser.add_argument('--opts', help='see cfgs/heatmap_cfgs_template.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    assert args.config_file is not None, "Please provide heatmap config file for parameters."
    cfg = cfg_utils.load_cfg_from_cfg_file(args.config_file)
    if args.opts is not None:
        cfg = cfg_utils.merge_cfg_from_list(cfg, args.opts)

    print(f"[*(//@_@)*]@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@[*(//@_@)*] HEATMAP CONFIG PARAMS:\n{cfg}")
    return cfg


def infer_single_slide(model, features, label, reverse_label_dict, k=1, prototype_feat=None):
	features = features.to(device)
	with torch.no_grad():
		if isinstance(model, (CLAM_SB, CLAM_MB, ABMIL)):
			logits, Y_prob, Y_hat, A, results_dict = model(features, train=False)
		elif isinstance(model, PhiHER2model):
			model.inst_num_twice = None
			logits, Y_prob, Y_hat, A, results_dict = model(features, prototype=prototype_feat)

		Y_hat = Y_hat.item()

		if isinstance(model, (CLAM_MB,)):
			A = A[Y_hat]

		A = A.view(-1, 1).cpu().numpy()


		print('Y_hat: {}, Y_label: {}, Y_prob: {}'.format(reverse_label_dict[Y_hat], label, ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]))	
		
		probs, ids = torch.topk(Y_prob, k)
		probs = probs[-1].cpu().numpy()
		ids = ids[-1].cpu().numpy()
		preds_str = np.array([reverse_label_dict[idx] for idx in ids])

	return ids, preds_str, probs, A


def load_params(df_entry, params):
	for key in params.keys():
		if key in df_entry.index:
			dtype = type(params[key])
			val = df_entry[key] 
			val = dtype(val)
			if isinstance(val, str):
				if len(val) > 0:
					params[key] = val
			elif not np.isnan(val):
				params[key] = val
			else:
				pdb.set_trace()

	return params


if __name__ == '__main__':
	args = get_parser()

	step_size = tuple((np.array(args.patch_size) * (1-args.overlap)).astype(int))
	print('patch_size: {}, with {:.2f} overlap, step size is {}'.format(args.patch_size, args.overlap, step_size))

	
	preset = args.preset
	def_seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': True, 
					  'keep_ids': 'none', 'exclude_ids':'none'}
	def_filter_params = {'a_t':13.0, 'a_h': 8.0, 'max_n_holes':10} # 最大倍率下512*512大小的个数所占的面积
	def_vis_params = {'vis_level': -1, 'line_thickness': 25, 'draw_grid': True}
	def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

	if preset is not None:
		preset_df = pd.read_csv(preset)
		for key in def_seg_params.keys():
			def_seg_params[key] = preset_df.loc[0, key]

		for key in def_filter_params.keys():
			def_filter_params[key] = preset_df.loc[0, key]

		for key in def_vis_params.keys():
			def_vis_params[key] = preset_df.loc[0, key]

		for key in def_patch_params.keys():
			def_patch_params[key] = preset_df.loc[0, key]


	if args.process_list is None:
		if isinstance(args.data_dir, list):
			slides = []
			for data_dir in args.data_dir:
				slides.extend(os.listdir(data_dir))
		else:
			slides = sorted(os.listdir(args.data_dir))
		
		if args.spec_slide is None:
			slides = [slide for slide in slides if args.slide_ext in slide] 
		elif isinstance(args.spec_slide, str) and os.path.exists(args.spec_slide):
			slides = pd.read_csv(args.spec_slide)
			if 'HER2status' in slides.columns:
				slides["label"] = slides['HER2status'].map(args.label_dict)
			elif 'test' in slides.columns:
				slides = slides['test'].dropna().tolist() # only get the test set samples for inference
			
		elif isinstance(args.spec_slide, list):
			slides = args.spec_slide
			
		df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)
		
	else:
		df = pd.read_csv(args.process_list)
		df = initialize_df(df, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)

	mask = df['process'] == 1
	process_stack = df[mask].reset_index(drop=True)
	total = len(process_stack)
	print('\nlist of slides to process: ')
	print(process_stack.head(len(process_stack)))

	print('\ninitializing model and prototypes from checkpoint')
	device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
	prototype_data = torch.load(args.cluster_path)
	global_cents_feats = prototype_data["global_centroid_feats"]
	print("Estimate number of cluster (GLOBAL): {}".format(len(global_cents_feats)))
	global_cents_feats = global_cents_feats.to(device)
	args.num_cluster = len(global_cents_feats)
	
	ckpt_path = args.ckpt_path
	print('\nckpt path: {}'.format(ckpt_path))
	
	if args.initiate_fn == 'initiate_model':
		model =  initiate_model(args, ckpt_path)
		model = model.to(device=device)	

	else:
		raise NotImplementedError

	if args.retccl_filepath is None:
		from models.resnet_custom import resnet50_baseline
		feature_extractor = resnet50_baseline(pretrained=True)

    # elif args.retccl_filepath.split('/')[-1] == "CCL_best_ckpt.pth":
	else:
		pretext_model = torch.load(args.retccl_filepath)
		
		from models.resnet_RetCCL import resnet50
		feature_extractor = resnet50(num_classes=2,mlp=False, two_branch=False, normlinear=True) # num_classes is random, that's fine. because we will: model.fc = nn.Identity()
		feature_extractor.fc = nn.Identity()
		feature_extractor.load_state_dict(pretext_model, strict=True)
		

	feature_extractor.eval()
	print('Done!')

	label_dict =  args.label_dict
	class_labels = list(label_dict.keys())
	class_encodings = list(label_dict.values())
	reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))} 

	if torch.cuda.device_count() > 1:
		device_ids = list(range(torch.cuda.device_count()))
		feature_extractor = nn.DataParallel(feature_extractor, device_ids=device_ids).to('cuda:0')
	else:
		feature_extractor = feature_extractor.to(device)

	os.makedirs(args.production_save_dir, exist_ok=True)
	os.makedirs(args.raw_save_dir, exist_ok=True)
	blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': args.patch_size, 'step_size': args.patch_size, 
	'custom_downsample':args.custom_downsample, 'level': args.patch_level, 'use_center_shift': args.use_center_shift}

	for i in range(len(process_stack)):
		slide_name = str(process_stack.loc[i, 'slide_id'])
		if args.slide_ext not in slide_name:
			slide_name+=args.slide_ext
		print('\nprocessing: ', slide_name)	

		try:
			label = process_stack.loc[i, 'label']
		except KeyError:
			label = 'Unspecified'

		slide_id = slide_name.replace(args.slide_ext, '')

		if not isinstance(label, str):
			grouping = reverse_label_dict[label]
		else:
			grouping = label

		p_slide_save_dir = os.path.join(args.production_save_dir, args.save_exp_code, str(grouping))
		os.makedirs(p_slide_save_dir, exist_ok=True)

		r_slide_save_dir = os.path.join(args.raw_save_dir, args.save_exp_code, str(grouping),  slide_id)
		os.makedirs(r_slide_save_dir, exist_ok=True)

		if args.use_roi:
			x1, x2 = process_stack.loc[i, 'x1'], process_stack.loc[i, 'x2']
			y1, y2 = process_stack.loc[i, 'y1'], process_stack.loc[i, 'y2']
			top_left = (int(x1), int(y1))
			bot_right = (int(x2), int(y2))
		else:
			top_left = None
			bot_right = None
		
		print('slide id: ', slide_id)
		print('top left: ', top_left, ' bot right: ', bot_right)

		if isinstance(args.data_dir, str):
			slide_path = os.path.join(args.data_dir, slide_name)
		elif isinstance(args.data_dir, dict):
			data_dir_key = process_stack.loc[i, args.data_dir_key]
			slide_path = os.path.join(args.data_dir[data_dir_key], slide_name)
		else:
			raise NotImplementedError

		mask_file = os.path.join(r_slide_save_dir, slide_id+'_mask.pkl')
		
		# Load segmentation and filter parameters
		seg_params = def_seg_params.copy()
		filter_params = def_filter_params.copy()
		vis_params = def_vis_params.copy()

		seg_params = load_params(process_stack.loc[i], seg_params)
		filter_params = load_params(process_stack.loc[i], filter_params)
		vis_params = load_params(process_stack.loc[i], vis_params)

		keep_ids = str(seg_params['keep_ids'])
		if len(keep_ids) > 0 and keep_ids != 'none':
			seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
		else:
			seg_params['keep_ids'] = []

		exclude_ids = str(seg_params['exclude_ids'])
		if len(exclude_ids) > 0 and exclude_ids != 'none':
			seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
		else:
			seg_params['exclude_ids'] = []

		for key, val in seg_params.items():
			print('{}: {}'.format(key, val))

		for key, val in filter_params.items():
			print('{}: {}'.format(key, val))

		for key, val in vis_params.items():
			print('{}: {}'.format(key, val))
		
		print('Initializing WSI object')
		wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
		print('Done!')

		wsi_ref_downsample = wsi_object.level_downsamples[args.patch_level]

		# the actual patch size for heatmap visualization should be the patch size * downsample factor * custom downsample factor
		vis_patch_size = tuple((np.array(args.patch_size) * np.array(wsi_ref_downsample) * args.custom_downsample).astype(int))

		block_map_save_path = os.path.join(r_slide_save_dir, '{}_blockmap.h5'.format(slide_id))
		mask_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(slide_id))
		if vis_params['vis_level'] < 0:
			best_level = wsi_object.wsi.get_best_level_for_downsample(16)
			vis_params['vis_level'] = best_level
		mask = wsi_object.visWSI(**vis_params, number_contours=True)
		mask.save(mask_path)
		
		features_path = os.path.join(r_slide_save_dir, slide_id+'.pt')
		h5_path = os.path.join(r_slide_save_dir, slide_id+'.h5')
	

		##### check if h5_features_file exists ######
		if not os.path.isfile(h5_path) :
			_, _, wsi_object = compute_from_patches(wsi_object=wsi_object, 
											model=model, prototype_feat=global_cents_feats,
											feature_extractor=feature_extractor, 
											batch_size=args.batch_size, **blocky_wsi_kwargs, 
											attn_save_path=None, feat_save_path=h5_path, 
											ref_scores=None)				
		
		##### check if pt_features_file exists ######
		if not os.path.isfile(features_path):
			file = h5py.File(h5_path, "r")
			features = torch.tensor(file['features'][:])
			torch.save(features, features_path)
			file.close()

		# load features 
		features = torch.load(features_path)
		process_stack.loc[i, 'bag_size'] = len(features)
		
		wsi_object.saveSegmentation(mask_file)
		Y_hats, Y_hats_str, Y_probs, A = infer_single_slide(model, features, label, reverse_label_dict, args.n_classes,
													  prototype_feat=global_cents_feats)
		del features
		
		if not os.path.isfile(block_map_save_path): 
			file = h5py.File(h5_path, "r")
			coords = file['coords'][:]
			file.close()
			asset_dict = {'attention_scores': A, 'coords': coords}
			block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')
		
		# save top 3 predictions
		for c in range(args.n_classes):
			process_stack.loc[i, 'Pred_{}'.format(c)] = Y_hats_str[c]
			process_stack.loc[i, 'prob_{}'.format(c)] = Y_probs[c]

		# os.makedirs('heatmaps/results/', exist_ok=True)
		if args.process_list is not None:
			process_stack.to_csv(os.path.join(args.raw_save_dir, '{}.csv'.format(args.process_list.replace('.csv', ''))), index=False)
		else:
			process_stack.to_csv(os.path.join(args.raw_save_dir, '{}.csv'.format(args.save_exp_code)), index=False)
		
		file = h5py.File(block_map_save_path, 'r')
		dset = file['attention_scores']
		coord_dset = file['coords']
		scores = dset[:]
		coords = coord_dset[:]
		file.close()

		samples = args.samples
		for sample in samples:
			if sample['sample']:
				tag = "label_{}_pred_{}".format(label, Y_hats[0])
				sample_save_dir =  os.path.join(args.production_save_dir, args.save_exp_code, 'sampled_patches', str(tag), sample['name'])
				os.makedirs(sample_save_dir, exist_ok=True)
				print('sampling {}'.format(sample['name']))
				sample_results = sample_rois(scores, coords, k=sample['k'], mode=sample['mode'], seed=sample['seed'], 
					score_start=sample.get('score_start', 0), score_end=sample.get('score_end', 1))
				for idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
					print('coord: {} score: {:.3f}'.format(s_coord, s_score))
					patch = wsi_object.wsi.read_region(tuple(s_coord), args.patch_level, args.patch_size).convert('RGB')
					patch.save(os.path.join(sample_save_dir, '{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, slide_id, s_coord[0], s_coord[1], s_score)))

		wsi_kwargs = {'top_left': top_left, 'bot_right': bot_right, 'patch_size': args.patch_size, 'step_size': step_size, 
		'custom_downsample':args.custom_downsample, 'level': args.patch_level, 'use_center_shift': args.use_center_shift}

		heatmap_save_name = '{}_blockmap.tiff'.format(slide_id)
		if os.path.isfile(os.path.join(r_slide_save_dir, heatmap_save_name)):
			pass
		else:
			heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, cmap=args.cmap, alpha=args.alpha, use_holes=True, binarize=False, vis_level=-1, blank_canvas=False,
							thresh=-1, patch_size = vis_patch_size, convert_to_percentiles=True)
		
			heatmap.save(os.path.join(r_slide_save_dir, '{}_blockmap.png'.format(slide_id)))
			del heatmap

		save_path = os.path.join(r_slide_save_dir, '{}_{}_roi_{}.h5'.format(slide_id, args.overlap, args.use_roi))

		if args.use_ref_scores:
			ref_scores = scores
		else:
			ref_scores = None
		
		if args.calc_heatmap:
			compute_from_patches(wsi_object=wsi_object, clam_pred=Y_hats[0], model=model, prototype_feat=global_cents_feats,
						feature_extractor=feature_extractor, batch_size=args.batch_size, **wsi_kwargs, 
								attn_save_path=save_path,  ref_scores=ref_scores)

		if not os.path.isfile(save_path):
			print('heatmap {} not found'.format(save_path))
			if args.use_roi:
				save_path_full = os.path.join(r_slide_save_dir, '{}_{}_roi_False.h5'.format(slide_id, args.overlap))
				print('found heatmap for whole slide')
				save_path = save_path_full
			else:
				continue

		file = h5py.File(save_path, 'r')
		dset = file['attention_scores']
		coord_dset = file['coords']
		scores = dset[:]
		coords = coord_dset[:]
		file.close()

		heatmap_vis_args = {'convert_to_percentiles': True, 'vis_level': args.vis_level, 'blur': args.blur, 'custom_downsample': args.custom_downsample}
		if args.use_ref_scores:
			heatmap_vis_args['convert_to_percentiles'] = False

		heatmap_save_name = '{}_{}_roi_{}_blur_{}_rs_{}_bc_{}_a_{}_l_{}_bi_{}_{}.{}'.format(slide_id, float(args.overlap), int(args.use_roi),
																						int(args.blur), 
																						int(args.use_ref_scores), int(args.blank_canvas), 
																						float(args.alpha), int(args.vis_level), 
																						int(args.binarize), float(args.binary_thresh), args.save_ext)


		if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
			pass
		
		else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
			heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object,  
						          cmap=args.cmap, alpha=args.alpha, **heatmap_vis_args, 
						          binarize=args.binarize, 
						  		  blank_canvas=args.blank_canvas,
						  		  thresh=args.binary_thresh,  patch_size = vis_patch_size,
						  		  overlap=args.overlap, 
						  		  top_left=top_left, bot_right = bot_right)
			if args.save_ext == 'jpg':
				heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
			else:
				heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))
		
		if args.save_orig:
			if args.vis_level >= 0:
				vis_level = args.vis_level
			else:
				vis_level = vis_params['vis_level']
			heatmap_save_name = '{}_orig_{}.{}'.format(slide_id,int(vis_level), args.save_ext)
			if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
				pass
			else:
				heatmap = wsi_object.visWSI(vis_level=vis_level, view_slide_only=True, custom_downsample=args.custom_downsample)
				if args.save_ext == 'jpg':
					heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
				else:
					heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))

	with open(os.path.join(args.raw_save_dir, args.save_exp_code, 'config.yaml'), 'w') as outfile:
		yaml.dump(args, outfile, default_flow_style=False)


