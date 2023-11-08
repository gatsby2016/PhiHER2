import os, time, argparse, h5py
from tqdm import tqdm
import numpy as np

import openslide

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP

from utils.file_utils import save_hdf5, stat_feat_patch_num
from utils.utils import print_network, collate_features
from utils.plip_zeroshot_utils import PLIP_ZeroShot


"""
class: features extraction from patches
"""
class featsExtraction(object):
    def __init__(self, feat_to_dir, csv_path, h5_dir, slide_dir, retccl_filepath=None, 
                 slide_ext="", auto_skip=False):
        """
        feat_to_dir:    dir for saving extracted feats
        csv_path:       csv file with slide_id column
        h5_dir:         DIR path to folder containing masks/ patches/ stitches/ subfolders
        slide_dir:      DIR for h5 files data
        retccl_filepath:None, use resnet50 pretrained model; path if SET, use RetCCL method for feature extraction
        slide_ext:      slide image suffix extension
        auto_skip
        """
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        assert feat_to_dir is not None, f"directory to save extracted feats data {feat_to_dir}"
        self.featdirdicts = self.create_featsubdirs(feat_to_dir)

        if auto_skip:
            dest_files = os.listdir(self.featdirdicts["pt_feats_subdir"]) # only auto_skip is True, dest files is used
        else:
            dest_files = None
        self.dest_files = dest_files

        # assert csv_path is not None or h5_dir is not None, f"Dir containing coordinate h5 files {h5_dir} or csvpath {csv_path} must be given :)"
        # print("[*(//@_@)*]@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@[*(//@_@)*]")
        assert csv_path is not None and os.path.isfile(csv_path), f"csvpath {csv_path} must be given or file do not EXIST :)"
        print('initializing dataset')
        self.bags_dataset = Dataset_All_Bags(csv_path)
        
        assert retccl_filepath is None or os.path.exists(retccl_filepath), f"{retccl_filepath} is not None, or file DO NOT EXIST."
        self.model = self.load_model(retccl_filepath, device=self.device)

        assert h5_dir is not None and os.path.exists(h5_dir), f"path to folder containing masks/ patches/ stitches/ subfolders {h5_dir} is None or do not EXIST =_=!"
        self.h5_dir = h5_dir
        assert slide_dir is not None and os.path.exists(slide_dir), f"path to folder containing raw slide images {slide_dir} is None or do not EXIST =_=!"        
        self.slide_dir = slide_dir

        assert slide_ext in ['.svs', '.mrxs', '.ndpi'], f"{slide_ext} should be in ['.svs', '.mrxs', '.ndpi']"
        self.slide_ext = slide_ext

    def run(self, batch_size=256, custom_downsample=1, gaussian_blur=False, resize_size=None, target_patch_size=-1, float16 = False, plip_tumor=False):
        """
        operate feats extraction
        batch_size:
        custom_downsample: 相较于WsiPatching中指定尺寸的下采样倍率 如WsiPatching中create patch为256, 这里指定2 则实际提取的patch为256然后resize到256//2=128进行特征提取
        target_patch_size: 目标patch size 如和WsiPatching中指定尺寸不一致 会resize处理为这个targetsize 不指定target size和downsample时默认不resize，即用create的patch size
        """
        if plip_tumor:
            def_types = ["tumor", "adipose", "stroma", "immune infiltrates lymphocytes", "gland", "necrosis or hemorrhage", "background or black", "non"]
            plip_model = PLIP_ZeroShot(model_path="/home/cyyan/Projects/HER2proj/scripts/plip/models/",
                                       types_text=def_types,
                                       device=self.device)
        else:
            plip_model = None
        
        total = len(self.bags_dataset)

        for bag_candidate_idx in range(total):
            print('\nprogress: {}/{}'.format(bag_candidate_idx, total))

            slide_id = self.bags_dataset[bag_candidate_idx].split(self.slide_ext)[0]
            print(slide_id)

            if self.dest_files is not None and slide_id+'.pt' in self.dest_files:
                print('skipped {}'.format(slide_id))
                continue

            h5_file_path = os.path.join(self.h5_dir, 'patches', slide_id+'.h5')
            if not os.path.exists(h5_file_path): # fix BUG: in case of some svs files lack of useful foreground
                print(f"{h5_file_path} do not exist. It may lack of foreground tissue regions. So skip.")
                continue

            slide_file_path = os.path.join(self.slide_dir, slide_id+self.slide_ext)
            output_path = os.path.join(self.featdirdicts['h5_feats_subdir'], slide_id+'.h5')
            
            time_start = time.time()
            self.compute_w_loader(h5_file_path, slide_file_path, output_path,
                        model = self.model, batch_size = batch_size, verbose = 1, 
                        gaussian_blur=gaussian_blur, resize_size=resize_size,
                        custom_downsample=custom_downsample, target_patch_size=target_patch_size,
                        plip_model=plip_model, device=self.device)
            time_elapsed = time.time() - time_start
            print('\ncomputing features for {} took {} s'.format(output_path, time_elapsed))
            
            file = h5py.File(output_path, "r")

            features = file['features'][:]
            print('features size: ', features.shape)
            # print('coordinates size: ', file['coords'].shape)

            # 保持的h5数据中存在coord为[-1,-1]标记，即对应patch损坏，feats仅用来置位，因此在保持无coord的pt数据时需要去掉这些feats
            features = features[file['coords'][:, 0] != -1, :] # 仅保留非-1标记的patch对应特征
            print('After filtering by coords MARKING; features size: ', features.shape)
            if plip_tumor:
                plip_tissue_idx = file['plip_tissue_idx'][:][file['coords'][:, 0] != -1]
                reserve_tissue_flag = [each in [0, 4] for each in plip_tissue_idx] # select specific tissue. 0 tumor, 4 gland 
                features = features[reserve_tissue_flag, :]
                print('After ONLY selecting TUMOR tiles; features size: ', features.shape)

            features = torch.from_numpy(features)
            if float16:
                features = features.type(torch.float16)
            torch.save(features, os.path.join(self.featdirdicts['pt_feats_subdir'], slide_id+'.pt'))

            if plip_tumor: # here we also save the plip feats in pt files    
                plip_feats = file['plip_feats'][:]
                plip_feats = plip_feats[file['coords'][:, 0] != -1, :]
                plip_feats = plip_feats[reserve_tissue_flag, :]
                
                plip_feats = torch.from_numpy(plip_feats)
                if float16:
                    plip_feats = plip_feats.type(torch.float16)
                os.makedirs(self.featdirdicts['pt_feats_subdir']+"_plip", exist_ok=True)
                torch.save(plip_feats, os.path.join(self.featdirdicts['pt_feats_subdir']+"_plip", slide_id+'.pt'))


    def stat_patch_num(self):
        stat_feat_patch_num(feat_dir=self.featdirdicts['feat_to_dir'], to_csv=True)

    @staticmethod
    def create_featsubdirs(feat_to_dir):
        """
        create subdirs by feat_to_dir
        """
        pt_feats_subdir = os.path.join(feat_to_dir, 'pt_files')
        h5_feats_subdir = os.path.join(feat_to_dir, 'h5_files')

        dirsdict = {'feat_to_dir': feat_to_dir,
                    'pt_feats_subdir': pt_feats_subdir, 
                    'h5_feats_subdir' : h5_feats_subdir} 
        
        for key, val in dirsdict.items():
            print("mkdir {} : {}".format(key, val))
            os.makedirs(val, exist_ok=True)

        return dirsdict

    @staticmethod
    def load_model(retccl_filepath=None, device="cuda"):
        """
        load model from retccl filepath or imagenet pretrained resnet50 model
        """
        print('loading model checkpoint...')
        if retccl_filepath is None:
            from models.resnet_custom import resnet50_baseline
            model = resnet50_baseline(pretrained=True)

        elif retccl_filepath.split('/')[-1] == "CCL_best_ckpt.pth":
            pretext_model = torch.load(retccl_filepath)

            from models.resnet_RetCCL import resnet50
            model = resnet50(num_classes=2,mlp=False, two_branch=False, normlinear=True) # num_classes is random, that's fine. because we will: model.fc = nn.Identity()
            model.fc = nn.Identity()
            model.load_state_dict(pretext_model, strict=True)

        elif retccl_filepath.split('/')[-1] == "ctranspath.pth":            
            from models.model_swinTrans import ctranspath
            model = ctranspath()
            model.head = nn.Identity()

            pretext_model = torch.load(retccl_filepath)
            model.load_state_dict(pretext_model['model'], strict=True)

        model = model.to(device)
                
        # print_network(model)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.eval()

        return model

    @staticmethod
    def compute_w_loader(file_path, slidewsi_path, output_path, model,
        batch_size = 8, verbose = 0, pretrained=True, gaussian_blur=False, resize_size=None,
        custom_downsample=1, target_patch_size=-1, plip_model=None, device="cuda"):
        """
        args:
            file_path: directory of bag (.h5 file)
            output_path: directory to save computed features (.h5 file)
            model: pytorch model
            batch_size: batch_size for computing features in batches
            verbose: level of feedback
            pretrained: use weights pretrained on imagenet
            gaussian_blur: use  gaussian_blur or not, default False
            custom_downsample: custom defined downscale factor of image patches
            target_patch_size: custom defined, rescaled image size before embedding
        """
        wsi = openslide.open_slide(slidewsi_path)

        dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, 
                                     gaussian_blur=gaussian_blur, resize_size = resize_size,
            custom_downsample=custom_downsample, target_patch_size=target_patch_size)
        # x, y = dataset[0]
        kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
        loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

        if verbose > 0:
            print('processing {}: total of {} batches'.format(file_path,len(loader)))

        mode = 'w'
        for batch, coords, batch_4plip in tqdm(loader, total=len(loader)):
            with torch.no_grad():	
                batch = batch.to(device, non_blocking=True)
                
                features = model(batch)
                features = features.cpu().numpy()

                if plip_model is not None:
                    plip_feats, tissue_type_idx = plip_model(batch_4plip)
                    plip_feats = plip_feats.cpu().numpy()
                    tissue_type_idx = tissue_type_idx.cpu().numpy()
                else:
                    plip_feats = np.array([-1])
                    tissue_type_idx = np.array([0]*len(batch_4plip))

                asset_dict = {'features': features, 'coords': coords, 'plip_feats': plip_feats, 'plip_tissue_idx': tissue_type_idx}
                save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
                mode = 'a'
        

def set_args():
    parser = argparse.ArgumentParser(description='Features Extraction')
    parser.add_argument('--h5_dir', type=str, default=None, help='(better absolute) DIR for h5 files data')
    parser.add_argument('--slide_dir', type=str, default=None, help='(better absolute) DIR for raw image slides')

    parser.add_argument('--csv_path', type=str, default=None, help='csv file with slide_id column')
    parser.add_argument('--retccl_filepath', type=str, default=None, help='if SET, use RetCCL method for feature extraction')
    
    parser.add_argument('--feat_to_dir', type=str, default=None, help='dir for saving extracted feats')

    parser.add_argument('--slide_ext', type=str, default= '.svs', help='slide image suffix extension')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--custom_downsample', type=int, default=1, 
                help='相较于createPatches函数中指定尺寸的下采样倍率; 如create patch为256, 这里指定2，则实际提取的patch为256然后resize到256//2=128进行特征提取')
    parser.add_argument('--target_patch_size', type=int, default=-1,
                help='目标patch size，如和createPatches函数中指定尺寸不一致，会resize处理为这个targetsize；不指定target size和downsample时默认不resize，即用create的patch size')
    # parser.add_argument('--no_auto_skip', default=False, action='store_true')
    parser.add_argument('--resize_size', type=int, default=None)

    parser.add_argument('--gaussian_blur', default=False, action='store_true')
    parser.add_argument('--auto_skip', default=False, action='store_true')
    parser.add_argument('--float16', default=False, action='store_true')
    parser.add_argument('--plip_tumor', default=False, action='store_true')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = set_args()

    # args.feat_to_dir = "/home/cyyan/Projects/HER2proj/results/tmp"
    # args.csv_path = "/home/cyyan/Projects/HER2proj/results/CLAMpatches/process_list_autogen.csv"
    # args.slide_dir = "/home/cyyan/Projects/HER2proj/data/reorganize/"
    # args.h5_dir = "/home/cyyan/Projects/HER2proj/results/CLAMpatches"
    # args.slide_ext = ".mrxs"
    # args.auto_skip = False
    # args.retccl_filepath = "/home/cyyan/Projects/HER2proj/data/CCL_best_ckpt.pth"
    # args.float16 = True

    featsextract = featsExtraction(args.feat_to_dir,
                                   args.csv_path, args.h5_dir, args.slide_dir, 
                                   args.retccl_filepath, 
                                   args.slide_ext, args.auto_skip)

    featsextract.run(args.batch_size, args.custom_downsample, args.gaussian_blur, args.resize_size,
                     args.target_patch_size, args.float16, args.plip_tumor)

    featsextract.stat_patch_num()