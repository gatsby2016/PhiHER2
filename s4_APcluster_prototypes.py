import os, json, argparse
import torch
import numpy as np
import h5py
import openslide

from datasets.dataset_generic import Generic_MIL_Dataset

import utils.config_utils as cfg_utils
from utils.general_utils import set_seed_torch
from utils.train_utils import _get_splits, _init_loaders
from utils.cluster_utils import local_apcluster, global_set_apcluster



def get_parser():
    parser = argparse.ArgumentParser(description='APcluster for prototypes script')
    parser.add_argument('--config_file', type=str, default=None) # "scripts/HE2HER2/cfgs/HEROHE.yaml")
    parser.add_argument('--opts', help='see cfgs/HEROHE.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    assert args.config_file is not None, "Please provide config file for parameters."
    cfg = cfg_utils.load_cfg_from_cfg_file(args.config_file)
    if args.opts is not None:
        cfg = cfg_utils.merge_cfg_from_list(cfg, args.opts)

    print(f"[*(//@_@)*]@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@[*(//@_@)*] CONFIG PARAMS:\n{cfg}")
    return cfg


"""
apcluster for all feat.pt files in `feat_root_dir`, save to `cluster_to_path`
default apcluster on feat num: min(10000, len(feats))
be careful of np.random HERE
"""
def loop_apcluster_folder(feat_root_dir, cluster_to_path, max_num=5000, **kwargs):
    os.makedirs(cluster_to_path, exist_ok=True)

    feat_files = os.listdir(feat_root_dir)
    feat_files = sorted([int(x.split(".")[0]) for x in feat_files]) # sorted as int(val)

    for feat_name in feat_files:
        feat_name = str(feat_name)+".pt"

        full_path = os.path.join(feat_root_dir, feat_name)
        feats = torch.load(full_path)

        # 下面加了一行进行slide层面patch的随机采样
        random_index = np.random.choice(len(feats), max_num) if len(feats) > max_num else np.arange(len(feats))
        
        local_cent_indices, local_cent_feat, usetime = local_apcluster(feats[random_index, :], **kwargs)

        assert torch.all(local_cent_feat - feats[random_index[local_cent_indices], ]==0.), f"{feat_name} do not align."
        
        print("AP cluster for prototypes, {}, Use time: {:.2f}\t Number of cluster: {}".format(\
            feat_name, usetime, len(local_cent_indices)))        
        torch.save({'centroid': local_cent_indices, 'centroid_feat': local_cent_feat, 
                    "feat_idx": random_index[local_cent_indices]},
                   os.path.join(cluster_to_path, feat_name))


"""
global apcluster training loader split, global cluster results are saved to `cluster_path`,
which load local apcluster results from `os.path.join(cluster_path, subfolder)`
no random sampling operation HERE.
"""
def global_apcluster_split(train_loader, cluster_path, subfolder, timeidx, cur, 
                           wsi_slide_path, wsi_patch_path, 
                           **kwargs):
    all_local_cents_feats = torch.Tensor()
    all_local_cents_idx = np.array([]) # 保留idx和对应slide id name，进行回取
    all_slide_pos = []

    train_split = train_loader.dataset

    for idx in range(len(train_split)): # loader的方式已经random打乱，通过split的方式保持了原始数据的顺序；两者已对齐
        (_, _, slide_id) = train_split[idx]

        assert os.path.exists(os.path.join(cluster_path, subfolder, slide_id+'.pt')),\
            f"cluster for {slide_id} do not exist."
        
        data = torch.load(os.path.join(cluster_path, subfolder, slide_id+'.pt'))
        print("Loading, AP cluster for prototypes, {}, Number of cluster: {}".format(slide_id, len(data["centroid"])))
        _, local_cent_feat, feat_idx = data["centroid"], data["centroid_feat"], data['feat_idx']

        all_local_cents_feats = torch.cat((all_local_cents_feats, local_cent_feat), dim=0)
        all_local_cents_idx = np.concatenate((all_local_cents_idx, feat_idx))
        all_slide_pos.extend([slide_id]*len(feat_idx))

    if True:
        global_cents_indices, global_cents_feats, apmodel, usetime = global_set_apcluster(all_local_cents_feats, **kwargs)
    else:
        data = torch.load(os.path.join(cluster_path, f"time_{timeidx}_fold_{cur}_prototypes.pt"))
        global_cents_indices, global_cents_feats = data['global_centroid_indices'], data["global_centroid_feats"]
        print(f"Loading, [GLOBAL] AP cluster for prototypes, Number of cluster: {len(global_cents_feats)}")

    print("[Global] AP cluster on {} for prototypes, Use time: {:.2f}\t Number of cluster: {}".format(\
        len(all_local_cents_feats), usetime, len(global_cents_indices)))

    torch.save({'global_centroid_indices': global_cents_indices, 
                'global_centroid_feats': global_cents_feats,
                'cluster_model': apmodel},
                os.path.join(cluster_path, f"time_{timeidx}_fold_{cur}_prototypes.pt"))

    reserve_wsi_region(all_local_cents_idx[global_cents_indices], 
                       np.array(all_slide_pos)[global_cents_indices],
                       wsi_slide_path=wsi_slide_path,
                       wsi_patch_path=wsi_patch_path,
                       prototypes_to_path=os.path.join(cluster_path,  f"time_{timeidx}_fold_{cur}"))

    print(">>>>>>>>>>> Estimate number of cluster (GLOBAL) >>>>>>>>>>>>>>>>: ", len(global_cents_indices))



"""
reserve the clusterring region according to `feats_idx` and `corrd_slide_id` with wsi object
and save to `prototypes_to_path`
"""
def reserve_wsi_region(feats_idx, corrd_slide_id, wsi_slide_path=None, wsi_patch_path=None, 
                       prototypes_to_path=None):
    os.makedirs(prototypes_to_path, exist_ok=True)

    for idx in range(len(corrd_slide_id)):
        slide_id = corrd_slide_id[idx]
        wsi = openslide.open_slide(os.path.join(wsi_slide_path, slide_id+".mrxs"))
        slide_idx = int(feats_idx[idx])

        h5_file_path = os.path.join(wsi_patch_path, "patches", slide_id+'.h5')
        assert os.path.exists(h5_file_path), f"{h5_file_path} does not exist."

        with h5py.File(h5_file_path, "r") as f:
            dset = f['coords']
        
            for name, value in dset.attrs.items():
                print(name, value)

            patch_level = dset.attrs['patch_level']
            patch_size = dset.attrs['patch_size']
            custom_downsample = dset.attrs['downsample'][0]
            # length = len(dset)
            if custom_downsample > 1:
                patch_size = patch_size // custom_downsample
            
            coord = dset[slide_idx] # 修改频繁读写h5 file为读取一次，保留到类中

        img = wsi.read_region(coord, patch_level, (patch_size, patch_size)).convert('RGB')
            # except:
            #     print("openslide.lowlevel.OpenSlideError Not a JPEG file: starts with 0xc3 0xcf', Now idx: ", idx)
            #     img = Image.new(size=(self.patch_size, self.patch_size), mode="RGB", color=(0,0,0))
            #     coord = np.array([-1, -1]) # 修改coord 坐标为（-1,-1）进行标记
        
        img.save(os.path.join(prototypes_to_path, 
                              f"p{idx}_slide{slide_id}_idx{slide_idx}_x{coord[0]}_y{coord[1]}.png"))
        print(f"p{idx}_slide{slide_id}_idx{slide_idx}_x{coord[0]}_y{coord[1]}.png has been saved to {prototypes_to_path}")
        


if __name__ == "__main__":
    args = get_parser()   
    args.device = set_seed_torch(args.gpu, args.seed) # set random seed

    max_num = 5000 #set the max tiles for each slide

    ### apcluster for all feats.pt files in  args.data_root_dir
    loop_apcluster_folder(feat_root_dir=os.path.join(args.data_root_dir, "pt_files"), 
                          cluster_to_path=os.path.join(args.cluster_path, "local_files"), 
                          max_num=max_num,
                          lamb=args.lamb, preference=args.preference, damping=args.damping, seed=args.seed)

    tstart = 0 if args.t_start == -1 else args.t_start
    tend = args.times if args.t_end == -1 else args.t_end
    times = np.arange(tstart, tend) 

    for timeidx in times: # multi times loop

        start = 0 if args.k_start == -1 else args.k_start
        end = args.k if args.k_end == -1 else args.k_end
        folds = np.arange(start, end)

        for cur in folds:

            labels_dict = {'Negative':0, 'Positive':1}
            filter_dict = {"HER2status": ["Negative", "Positive"]}
            dataset_factory = Generic_MIL_Dataset(csv_path = args.csv_info_path,
                                    data_dir= args.data_root_dir,
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = labels_dict,
                                    filter_dict = filter_dict,
                                    label_col = args.label_col,
                                    patient_voting='maj', # maj合理；max是对于一个patient多个slide标签选标签值最大，不合理
                                    patient_strat= True, # TRUE 表示按照patient进行划分split，且保证同一个patient的所有slide被split在同一区间内，如train或val；否则直接按slideID进行split
                                    ignore=[],
                                    num_perslide=None,
                                    )

            print(f"Created train and val datasets for time {timeidx} fold {cur}")
            if args.split_dir.split("/")[-1][-12:] == "TrainValTest":
                csv_path = f'{args.split_dir}/splits_time{timeidx}.csv'
            else:
                csv_path = f'{args.split_dir}/splits_time{timeidx}_fold{cur}.csv'
                
            
            datasets = dataset_factory.return_splits(from_id=False, csv_path=csv_path)
            datasets[0].set_split_id(split_id=cur)
            datasets[1].set_split_id(split_id=cur)

            train_split, val_split, test_split = _get_splits(datasets, cur)
            train_loader, val_loader, test_loader = _init_loaders(args, train_split, val_split, test_split)

            print(f">>>>>>>>>>> APcluster_prototypes on split >>>>>>>>>>>>>>>> fold: {cur} on time: {timeidx}")
            global_apcluster_split(train_loader, args.cluster_path, "local_files", 
                                   timeidx, cur,
                                   wsi_slide_path = "/mnt/DATA/HEROHE_challenge/TrainSet/",
                                   wsi_patch_path = args.data_root_dir.replace("2FeatsCCL", "1WsiPatching"),
                                   lamb=args.lamb, preference=args.preference, 
                                   damping=args.damping, seed=args.seed) # only AP cluster prototypes for training dataset 

    print(f">>>>>>>>>>> Finished APcluster_prototypes >>>>>>>>>>>>>>>> {cur} folds / {timeidx} times")