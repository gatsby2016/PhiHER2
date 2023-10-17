# coding=utf-8
import cv2
import numpy as np
from collections import Counter
import time
# from numba import jit
import numpy as np
import cv2
from skimage.feature import graycomatrix
import skimage.measure as skm
from scipy.ndimage import convolve


# @jit(nopython=True)
def calcIJ(img_patch):
    total_p = img_patch.shape[0] * img_patch.shape[1]
    if total_p % 2 != 0:
        center_p = img_patch[int(img_patch.shape[0] / 2), int(img_patch.shape[1] / 2)]
        mean_p = (np.sum(img_patch) - center_p) / (total_p - 1)
        return (center_p, mean_p)
    else:
        pass


def calcEntropy2dSpeedUp(img, win_w=3, win_h=3):
    height = img.shape[0]

    ext_x = int(win_w / 2)
    ext_y = int(win_h / 2)

    ext_h_part = np.zeros([height, ext_x], img.dtype)
    tem_img = np.hstack((ext_h_part, img, ext_h_part))
    ext_v_part = np.zeros([ext_y, tem_img.shape[1]], img.dtype)
    final_img = np.vstack((ext_v_part, tem_img, ext_v_part))

    new_width = final_img.shape[1]
    new_height = final_img.shape[0]

    # 最耗时的步骤，遍历计算二元组
    IJ = []
    for i in range(ext_x, new_width - ext_x):
        for j in range(ext_y, new_height - ext_y):
            patch = final_img[j - ext_y:j + ext_y + 1, i - ext_x:i + ext_x + 1]
            ij = calcIJ(patch)
            IJ.append(ij)

    Fij = Counter(IJ).items()

    # 第二耗时的步骤，计算各二元组出现的概率
    Pij = []
    for item in Fij:
        Pij.append(item[1] * 1.0 / (new_height * new_width))

    H_tem = []
    for item in Pij:
        h_tem = -item * (np.log(item) / np.log(2))
        H_tem.append(h_tem)

    H = np.sum(H_tem)
    return H



def calcEntropy_skimage(img):
    glcm = np.squeeze(graycomatrix(img, distances=[1], 
                                angles=[0], symmetric=True, 
                                normed=True))
    entropy = -np.sum(glcm*np.log2(glcm + (glcm==0)))
    return entropy



def calEnergy(img):
	filter_du = np.array([
		[1.0, 2.0, 1.0],
		[0.0, 0.0, 0.0],
		[-1.0, -2.0, -1.0],
	])
	filter_du = np.stack([filter_du] * 3, axis=2)

	filter_dv = np.array([
		[1.0, 0.0, -1.0],
		[2.0, 0.0, -2.0],
		[1.0, 0.0, -1.0],
	])
	filter_dv = np.stack([filter_dv] * 3, axis=2)

	img = img.astype('float32')
	convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))
	# energy_maps = convolved.sum(axis=2)
	energy = convolved.sum()
	return energy

if __name__ == "__main__":
    import os

    imgpath = "/home/cyyan/Projects/HER2proj/results/Yale_4Heatmaps/heatmap_production/HEATMAP_OUTPUT/sampled_patches/label_Unspecified_pred_0/topk_high_attention/"
    imgname = os.listdir(imgpath)

    for name in imgname:
        
        img = cv2.imread(os.path.join(imgpath, name), 1)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        print(calEnergy(img))
        
        #下述等价skm.shannon_entropy
        # marg = np.histogramdd(np.ravel(img), bins = 256)[0]/img.size
        # marg = list(filter(lambda p: p > 0, np.ravel(marg)))
        # entropy = -np.sum(np.multiply(marg, np.log2(marg)))

        print(skm.shannon_entropy(img), "vs.", calcEntropy_skimage(img), "vs.", calcEntropy2dSpeedUp(img))

    print('')