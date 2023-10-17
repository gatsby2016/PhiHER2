import pickle
import h5py

def save_pkl(filename, save_object):
	writer = open(filename,'wb')
	pickle.dump(save_object, writer)
	writer.close()

def load_pkl(filename):
	loader = open(filename,'rb')
	file = pickle.load(loader)
	loader.close()
	return file


def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path


"""
因为在提取features时存在根据coord topleft坐标点无法read region的情况，所有有的patch特征是置位的，用coord为（-1，-1）标记；一起保存在单slide对应的h5文件中
该函数目的在于读取h5文件所在文件夹，然后统计队列整体的理想patch数量与实际提取到特征的patch数量
"""
def stat_feat_patch_num(feat_dir, to_csv = False):
    import os

    h5filelist = os.listdir(os.path.join(feat_dir, 'h5_files'))
    h5filelist = sorted(h5filelist)

    import pandas as pd
    import numpy as np

    total = len(h5filelist)
    default_df_dict = {'slide_id': np.full((total), 'to_be_added'),
                       'num_patch_coord': np.full((total), 1, dtype=np.uint32),
                       'num_patch_feats': np.full((total), 1, dtype=np.uint32)}
    # default_df_dict.update({'label': np.full((total), -1)})
    stat_slides = pd.DataFrame(default_df_dict)

    print_patcherror_slidelist = []
    for idx, filename in enumerate(h5filelist):
        print(filename)

        h5file_path = os.path.join(feat_dir, 'h5_files', filename)
        file = h5py.File(h5file_path, "r")

        num_patch_coord = len(file['coords'])
        print('(Ideal) Num of patch: ', num_patch_coord)
        num_patch_feats = sum(file['coords'][:, 0] != -1)
        print('(Actual) Num of patch: ', num_patch_feats)
        
        stat_slides.loc[idx, 'slide_id'] = filename.split('.')[0]
        stat_slides.loc[idx, 'num_patch_coord'] = num_patch_coord
        stat_slides.loc[idx, 'num_patch_feats'] = num_patch_feats
        
        if num_patch_feats != num_patch_coord:
            print_patcherror_slidelist.append(np.array([filename, num_patch_coord, num_patch_feats]))
    
    print(np.array(print_patcherror_slidelist))
    
    if to_csv:
        stat_slides.to_csv(os.path.join(feat_dir, 'slides_of_num_patch_feat_autogen.csv'), index=False)


if __name__ == "__main__":
    stat_feat_patch_num(feat_dir="/home/cyyan/Projects/HER2proj/results/CLAMfeats", to_csv=True)
