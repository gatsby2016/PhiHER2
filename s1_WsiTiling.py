# other imports
import os, time, argparse
import numpy as np
import pandas as pd

from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords


def initialize_df(slides, seg_params, filter_params, vis_params, patch_params, 
	use_heatmap_args=False, save_patches=False):
    """
    initiate a pandas df describing a list of slides to process

    args:
	slides (df or array-like): 
		array-like structure containing list of slide ids, if df, these ids assumed to be
		stored under the 'slide_id' column
	seg_params (dict): segmentation paramters 
	filter_params (dict): filter parameters
	vis_params (dict): visualization paramters
	patch_params (dict): patching paramters
	use_heatmap_args (bool): whether to include heatmap arguments such as ROI coordinates
    """
    total = len(slides)
    if isinstance(slides, pd.DataFrame):
        slide_ids = slides.slide_id.values
    else:
        slide_ids = slides
    default_df_dict = {'slide_id': slide_ids}

    # initiate empty labels in case not provided
    if use_heatmap_args:
        default_df_dict.update({'label': np.full((total), -1)})

    default_df_dict.update({
        'status': np.full((total), 'todo'),

        # 添加slide information
        'info_num_level': np.full((total), int(0), dtype=np.int8),
        'info_mpp': np.full((total), 0.0, dtype=np.float32), #
        'info_max_magn': np.full((total), int(0), dtype=np.int8), # 10 or 20 or 40
        'info_max_w': np.full((total), int(0), dtype=np.uint32),
        'info_max_h': np.full((total), int(0), dtype=np.uint32),

        # 添加seg and patching 后的foreground contour和holes number 信息，以及patch 信息
        'stat_seg_n_fore': np.full((total), int(-1), dtype=np.int8),
        'stat_seg_n_hole': np.full((total), int(-1), dtype=np.int8), # 该数字不超过max_n_holes
        'stat_patch_num': np.full((total), int(0), dtype=np.uint32),

        # bgtissue params
        'seg_level': np.full((total), int(seg_params['seg_level']), dtype=np.int8),
        'sthresh': np.full((total), int(seg_params['sthresh']), dtype=np.int8),
        'mthresh': np.full((total), int(seg_params['mthresh']), dtype=np.int8),
        'close': np.full((total), int(seg_params['close']), dtype=np.uint32),
        'use_otsu': np.full((total), bool(seg_params['use_otsu']), dtype=bool),
        'keep_ids': np.full((total), seg_params['keep_ids']),
        'exclude_ids': np.full((total), seg_params['exclude_ids']),
        
        # filter params
        'a_t': np.full((total), int(filter_params['a_t']), dtype=np.float32),
        'a_h': np.full((total), int(filter_params['a_h']), dtype=np.float32),
        'max_n_holes': np.full((total), int(filter_params['max_n_holes']), dtype=np.uint32),

        # vis params
        'vis_level': np.full((total), int(vis_params['vis_level']), dtype=np.int8),
        'line_thickness': np.full((total), int(vis_params['line_thickness']), dtype=np.uint32),
        'draw_grid': np.full((total), bool(vis_params['draw_grid']), dtype=bool),

        # patching params
        'patch_level': np.full((total), int(patch_params['patch_level']), dtype=np.uint8),
        'patch_size': np.full((total), int(patch_params['patch_size']), dtype=np.uint32),
        'step_size': np.full((total), int(patch_params['step_size']), dtype=np.uint32),
        'use_padding': np.full((total), bool(patch_params['use_padding']), dtype=bool),
        'contour_fn': np.full((total), patch_params['contour_fn'])
        })

    if save_patches:
        default_df_dict.update({
            'white_thresh': np.full((total), int(patch_params['white_thresh']), dtype=np.uint8),
            'black_thresh': np.full((total), int(patch_params['black_thresh']), dtype=np.uint8)})

    if use_heatmap_args:
        # initiate empty x,y coordinates in case not provided
        default_df_dict.update({'x1': np.empty((total)).fill(np.NaN), 
            'x2': np.empty((total)).fill(np.NaN), 
            'y1': np.empty((total)).fill(np.NaN), 
            'y2': np.empty((total)).fill(np.NaN)})


    if isinstance(slides, pd.DataFrame):
        temp_copy = pd.DataFrame(default_df_dict) # temporary dataframe w/ default params
        # find key in provided df
        # if exist, fill empty fields w/ default values, else, insert the default values as a new column
        for key in default_df_dict.keys(): 
            if key in slides.columns:
                naflag = slides[key].isna()
                slides.loc[naflag, key] = temp_copy.loc[naflag, key]
            else:
                slides.insert(len(slides.columns), key, default_df_dict[key])
    else:
        slides = pd.DataFrame(default_df_dict)

    return slides


"""
class: Patching from WSI
"""
class WsiPatching(object):

    def __init__(self, sourcepath, save_dir, process_list=None, 
                 preset=None, patch_level=0, patch_size=256, step_size=256,                  
                 patch=False, bgtissue=False, stitch=False, auto_skip=False):
        """
        sourcepath:     'path to folder containing raw wsi image files'
        save_dir:       'directory to save processed data'
        process_list:   'name of list of images to process with parameters (.csv)' fullfilename

        preset:         'fullfilename, predefined profile of default segmentation and filter parameters (.csv)'
        patch_level:    [default 0  ] 'downsample level at which to patch'
        patch_size:     [default 256] patch size
        step_size:      [default 256] sliding windows step

        patch:      [default False] patch flag
        bgtissue:   [default False] bgtissue flag
        stitch:     [default False] stitch flag
        auto_skip:  [default False] auto skip files if exist
        """

        assert sourcepath is not None, f"path to folder containing raw wsi image files {sourcepath} must be given :)"
        assert os.path.exists(sourcepath), f"path to folder containing raw wsi image files {sourcepath} do not EXIST =_=!"
        assert save_dir is not None, f"directory to save processed data {save_dir}"
        print("[*(//@_@)*]@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@[*(//@_@)*]")
        
        dirsdict = self.create_ressubdirs(save_dir=save_dir) # create save_dir and corresponding subdirs

        print('sourcepath: ', sourcepath)
        dirsdict.update({'sourcepath': sourcepath}) # update all dirs to dirsdict
        self.dirsdict = dirsdict

        assert process_list is None or os.path.exists(process_list), \
            f"process list: {process_list} do not EXIST. file of list of images to process(.csv) =_=!"
        self.process_list = process_list

        assert preset is None or os.path.exists(preset), f"preset list: {preset} do not EXIST. predefined profile of default bgtissue and filter parameters =_=!"
        preparams = self.set_params(preset) # set segmentation and filter preprocessing parameters in this func.
        preparams['patch_params'].update({'patch_level': patch_level, 
                                          'patch_size': patch_size, 
                                          'step_size': step_size})
        self.preparams = preparams

        self.patch = patch
        self.bgtissue = bgtissue
        self.stitch = stitch 
        self.auto_skip = auto_skip
        
    """
    create subdirs by save_dir
    """
    @staticmethod
    def create_ressubdirs(save_dir):
        patch_save_dir = os.path.join(save_dir, 'patches')
        mask_save_dir = os.path.join(save_dir, 'masks')
        stitch_save_dir = os.path.join(save_dir, 'stitches')
        print('patch_save_dir: ', patch_save_dir)
        print('mask_save_dir: ', mask_save_dir)
        print('stitch_save_dir: ', stitch_save_dir)

        dirsdict = {'save_dir': save_dir,
                    'patch_save_dir': patch_save_dir, 
                    'mask_save_dir' : mask_save_dir, 
                    'stitch_save_dir': stitch_save_dir} 
        
        for key, val in dirsdict.items():
            print("mkdir {} : {}".format(key, val))
            os.makedirs(val, exist_ok=True)

        return dirsdict
    
    """
    set profile of default segmentation and filter parameters
    """
    @staticmethod
    def set_params(preset):
        """
        close运算kernel为4x4 使用ostu大津阈值法True; max_n_holes 空洞数量修改为10
        ostu为true则不需要用到sthresh 只有simple binary thresholding才用到sthresh  
        修改patch params  use_padding 为false  
        增加vis params  'draw_grid': True  
        img转hsv，然后对色调（H），饱和度（S），明度（V）的饱和度空间， 进行中值滤波，进行大津阈值，
        
        sthresh: 阈值法 threshold，use_otsu为true时该参数无效
        mthresh: 中值滤波阈值
        close: 阈值法后，形态学运算，close operation的kernel 参数
        
        a_t: 去除掉前景区域的所有hole的面积和后的foreground面积阈值; 实际计算中filter_params['a_t'] * 512 / level_downsamples[seg_level]
        a_h: hole 面积阈值，小于该值的hole不考虑
        max_n_holes: 过滤holes，每个前景轮廓内最多保留max_n_holes个

        """
        seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': True,
                    'keep_ids': 'none', 'exclude_ids': 'none'}
        filter_params = {'a_t':30, 'a_h': 10, 'max_n_holes':10} #
        vis_params = {'vis_level': -1, 'line_thickness': 50, 'draw_grid': True}
        patch_params = {'use_padding': False, 'contour_fn': 'four_pt'}

        if preset:
            preset_df = pd.read_csv(preset)
            for key in seg_params.keys():
                seg_params[key] = preset_df.loc[0, key]

            for key in filter_params.keys():
                filter_params[key] = preset_df.loc[0, key]

            for key in vis_params.keys():
                vis_params[key] = preset_df.loc[0, key]

            for key in patch_params.keys():
                patch_params[key] = preset_df.loc[0, key]
        
        parameters = {'seg_params': seg_params,
                    'filter_params': filter_params,
                    'patch_params': patch_params,
                    'vis_params': vis_params}

        print(parameters)
        return parameters


    def run(self):
        seg_times, patch_times = self.seg_and_patch(**self.dirsdict, **self.preparams, 
                                                    bgtissue = self.bgtissue, 
                                                    stitch= self.stitch,
                                                    patch = self.patch,
                                                    auto_skip=self.auto_skip, process_list = self.process_list)


    def seg_and_patch(self, sourcepath, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir, 
                      seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False, 'keep_ids': 'none', 'exclude_ids': 'none'},
                      filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}, 
                      vis_params = {'vis_level': -1, 'line_thickness': 500, 'draw_grid': False},
                      patch_params = {'use_padding': True, 'contour_fn': 'four_pt', 'patch_level': 0, 'patch_size': 256, 'step_size': 256},
                      bgtissue = False, stitch= False, patch = False, auto_skip=False, process_list = None):
        
        slides = sorted(os.listdir(sourcepath))
        slides = [slide for slide in slides if os.path.isfile(os.path.join(sourcepath, slide))]
        if process_list is None:
            df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)
        
        else:
            df = pd.read_csv(process_list)
            df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

        mask = df['status'] != 'processed'
        process_stack = df[mask]

        total = len(process_stack)

        seg_times = 0.
        patch_times = 0.
        stitch_times = 0.

        for i in range(total):
            df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
            idx = process_stack.index[i]
            slide = process_stack.loc[idx, 'slide_id']
            print(f"{i:3d}/{total} Processing {slide}")
            
            slide_id, _ = os.path.splitext(slide)

            if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
                print('{} already exist in destination location, skipped'.format(slide_id))
                df.loc[idx, 'status'] = 'already_exist'
                continue

            # Inialize WSI
            WSI_object = WholeSlideImage(os.path.join(sourcepath, slide))

            current_filter_params = {}
            for key in filter_params.keys():
                current_filter_params.update({key: df.loc[idx, key]})

            current_patch_params = {}
            for key in patch_params.keys():
                current_patch_params.update({key: df.loc[idx, key]})
            current_patch_params.update({'save_path': patch_save_dir})

            current_seg_params = {}
            for key in seg_params.keys():
                current_seg_params.update({key: df.loc[idx, key]})

            current_vis_params = {}
            for key in vis_params.keys():
                current_vis_params.update({key: df.loc[idx, key]})

            if current_vis_params['vis_level'] < 0 or current_seg_params['seg_level'] < 0:
                if len(WSI_object.level_dim) == 1:
                    current_vis_params['vis_level'] = 0
                    current_seg_params['seg_level'] = 0
                else:	
                    wsi = WSI_object.getOpenSlide()
                    best_level = wsi.get_best_level_for_downsample(64)
                    current_vis_params['vis_level'] = best_level
                    current_seg_params['seg_level'] = best_level


            keep_ids = str(current_seg_params['keep_ids'])
            if keep_ids != 'none' and len(keep_ids) > 0:
                str_ids = current_seg_params['keep_ids']
                current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
            else:
                current_seg_params['keep_ids'] = []

            exclude_ids = str(current_seg_params['exclude_ids'])
            if exclude_ids != 'none' and len(exclude_ids) > 0:
                str_ids = current_seg_params['exclude_ids']
                current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
            else:
                current_seg_params['exclude_ids'] = []

            w, h = WSI_object.level_dim[current_seg_params['seg_level']] 
            if w * h > 1e8:
                print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
                df.loc[idx, 'status'] = 'failed_seg'
                continue

            seg_time_elapsed = -1
            if bgtissue:
                WSI_object, seg_time_elapsed = self.segment(WSI_object, current_seg_params, current_filter_params)

                mask = WSI_object.visWSI(**current_vis_params)
                mask_path = os.path.join(mask_save_dir, slide_id+'.png') # 保存图片为png格式，不用jpg
                mask.save(mask_path)

                # 20230314 将segment的contour和hole信息写入data frame 写入csv文件
                df.loc[idx, 'stat_seg_n_fore'] = len(WSI_object.contours_tissue)
                df.loc[idx, 'stat_seg_n_hole'] = sum([len(hole) for hole in WSI_object.holes_tissue])

            patch_time_elapsed = -1 # Default time
            if patch:
                file_path, patch_time_elapsed = self.patching(WSI_object = WSI_object,  **current_patch_params,)
            
            stitch_time_elapsed = -1
            if stitch:
                file_path = os.path.join(patch_save_dir, slide_id+'.h5')
                if os.path.isfile(file_path):
                    heatmap, patch_num, stitch_time_elapsed = self.stitching(file_path, WSI_object, downscale=64, draw_grid=current_vis_params['draw_grid'])
                    stitch_path = os.path.join(stitch_save_dir, slide_id+'.png') # 保存图片为png格式，不用jpg
                    heatmap.save(stitch_path)

                    df.loc[idx, 'stat_patch_num'] = patch_num

            df = self.update_dfinfo(df, idx, WSI_object, current_vis_params['vis_level'], current_seg_params['seg_level'])
            # print("segmentation took {} seconds".format(seg_time_elapsed))
            # print("patching took {} seconds".format(patch_time_elapsed))
            # print("stitching took {} seconds".format(stitch_time_elapsed))

            seg_times += seg_time_elapsed
            patch_times += patch_time_elapsed
            stitch_times += stitch_time_elapsed

        df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
        print("Total segmentation time: {} s".format(seg_times))
        print("Total patching time: {} s".format(patch_times))
        print("Total stiching time: {} s".format(stitch_times))
            
        return seg_times, patch_times

    @staticmethod
    def update_dfinfo(df, idx, WSI_object, vis_level, seg_level):
        """
        update dataframe info 
        """
        # 更新df中slide的information列
        df.loc[idx, 'vis_level'] = vis_level
        df.loc[idx, 'seg_level'] = seg_level

        df.loc[idx, 'info_num_level'] = len(WSI_object.level_dim) # or, WSI_object.wsi.level_count
        
        # BUG, fix bug: wsi.properties key `aperio.AppMag` and "aperio.MPP" and no such key for TCGA format file
        if "aperio.MPP" in WSI_object.wsi.properties:
            wsimpp = WSI_object.wsi.properties["aperio.MPP"]
            maxmag = WSI_object.wsi.properties["aperio.AppMag"]
        elif "openslide.mpp-x" in WSI_object.wsi.properties:
            wsimpp = WSI_object.wsi.properties["openslide.mpp-x"]
            maxmag = WSI_object.wsi.properties["openslide.objective-power"]
        else:
            wsimpp = 0
            maxmag = 0

        df.loc[idx, 'info_mpp'] = np.float32(wsimpp) # slide mpp
        df.loc[idx, 'info_max_magn'] = np.uint8(np.float32(maxmag)) # max magnification

        df.loc[idx, 'info_max_w'] = WSI_object.level_dim[0][0]
        df.loc[idx, 'info_max_h'] = WSI_object.level_dim[0][1]

        df.loc[idx, 'status'] = 'processed'

        return df

    @staticmethod
    def segment(WSI_object, seg_params = None, filter_params = None, mask_file = None):       
        start_time = time.time() ### Start bgtissue Timer
        if mask_file is not None: # Use segmentation file
            WSI_object.initSegmentation(mask_file)
        else: # Segment	
            WSI_object.segmentTissue(**seg_params, filter_params=filter_params)
        seg_time_elapsed = time.time() - start_time    ### Stop bgtissue Timers

        return WSI_object, seg_time_elapsed

    @staticmethod
    def patching(WSI_object, **kwargs):
        start_time = time.time() ### Start Patch Timer
        file_path = WSI_object.process_contours(**kwargs) # Patch
        patch_time_elapsed = time.time() - start_time ### Stop Patch Timer

        return file_path, patch_time_elapsed

    @staticmethod
    def stitching(file_path, wsi_object, downscale = 64, draw_grid=False):
        start_time = time.time()
        heatmap, patch_num = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0,0,0), alpha=-1, draw_grid=draw_grid)
        stitch_time_elapsed = time.time() - start_time
        
        return heatmap, patch_num, stitch_time_elapsed



def set_args():
    parser = argparse.ArgumentParser(description='WSI patching')
    parser.add_argument('-s', '--sourcepath', type = str,
                        help='path to folder containing raw wsi image files')    
    parser.add_argument('-d', '--save_dir', type = str,
                        help='directory to save processed data')
    parser.add_argument('-l', '--process_list',  type = str, default=None,
                        help='full path filename of list of images to process with parameters (.csv), if set')    
    
    parser.add_argument('-pl', '--patch_level', type=int, default=0, 
                        help='downsample level at which to patch')
    parser.add_argument('-ps', '--patch_size', type = int, default=256,
                        help='patch_size')
    parser.add_argument('-ss', '--step_size', type = int, default=256,
                        help='step_size')
    
    parser.add_argument('--patch', default=False, action='store_true')
    parser.add_argument('--bgtissue', default=False, action='store_true')
    parser.add_argument('--stitch', default=False, action='store_true')
    parser.add_argument('--auto_skip', default=False, action='store_true')

    parser.add_argument('--preset', type=str, default=None, 
                        help='predefined profile of default segmentation and filter parameters (.csv), if set, full filename path')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = set_args()
    # args.sourcepath = "/home/cyyan/Projects/HER2proj/data_ModPath_HER2_v3/pkg_v3/Yale_HER2_cohort/SVS"
    # args.save_dir = "/home/cyyan/Projects/HER2proj/results/WsiPatchingYale"

    patchinger = WsiPatching(sourcepath = args.sourcepath, save_dir = args.save_dir, 
                             process_list=args.process_list,
                             patch_level=args.patch_level, patch_size=args.patch_size, step_size=args.step_size, 
                             patch=args.patch, bgtissue=args.bgtissue, stitch=args.stitch, auto_skip=args.auto_skip, preset=args.preset)
    patchinger.run()
    
    # !python WsiPatching.py -s "/home/cyyan/Projects/HER2proj/data_ModPath_HER2_v3/pkg_v3/Yale_HER2_cohort/SVS" -d "/home/cyyan/Projects/HER2proj/results/WsiPatchingYale"