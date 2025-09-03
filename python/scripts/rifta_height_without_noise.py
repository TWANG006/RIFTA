import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
import sys
import time
sys.path.append( '../rifta')
from rifta_height import rifta_height
from show_rifta_height_estimation_result import show_rifta_height_estimation_result
from height_slope import height_2_slopes


def tic():
    global start_time
    start_time = time.time()
def toc():
    elapsed_time = time.time() - start_time
    print("Elapsed time: {:.8f} seconds".format(elapsed_time))


# # 定义数据目录和文件名
data_dir = '../../data/'
surf_file = 'sim_surf_with_slopes.mat'
# 构建完整的文件路径
full_path = os.path.join(data_dir, surf_file)
mat_data = loadmat(full_path)
# 定义窗口大小
winSz = 2.5e-3
X = mat_data['X']
Y = mat_data['Y']
Zf = mat_data['Zf']
pixel_m = np.median(np.diff(X[0, :]))#分辨率：[m/pixel]，米/像素
tifParams = {}
tifParams['A'] = 10e-9
tifParams['lat_res_tif'] = pixel_m
tifParams['d'] = 10e-3
tifParams['d_pix'] = round(tifParams['d'] / pixel_m)
tifParams['sigma_xy'] = [tifParams['d'] / 10, tifParams['d'] / 10]
min_x = np.nanmin(X)
min_y = np.nanmin(Y)
max_y = np.nanmax(Y)
ca_range_x = 190e-3
ca_range_y = 15e-3
ca_x_s = 15e-3
ca_y_s = 10e-3
ca_x_e = ca_x_s + ca_range_x
ca_y_e = ca_y_s + ca_range_y
ca_range = {}
ca_range['u_s'] = round((ca_x_s - min_x) / pixel_m)
ca_range['u_e'] = round((ca_x_e - min_x) / pixel_m)
ca_range['v_s'] = round((max_y - ca_y_e) / pixel_m)
ca_range['v_e'] = round((max_y - ca_y_s) / pixel_m)
use_tif_data = False #True、False
if use_tif_data:
    Xtif = mat_data['Xtif']
    Ytif = mat_data['Ytif']
    Ztif = mat_data['Ztif']
    options_h=None
else:
    Xtif = None
    Ytif = None
    Ztif = None
    options_h = {
    'algorithm': 'fft',
    'tifMode': 'model',
    'isResampling': False,
    'resamplingInterval': 1e-3,
    'ratio': 1,
    'maxIters': 20,
    'rmsDif': 0.01e-9,
    'dwellTimeDif': 30
    }

(X_B, Y_B,B,_, _,X_P, Y_P,T_P,_, _,_,_, _, _,Xca, Yca,Z_to_remove_ca, Z_removal_ca, Z_residual_ca) = rifta_height(X, Y, Zf, tifParams, Xtif, Ytif, Ztif, ca_range, options_h)
Zx_to_remove_ca, Zy_to_remove_ca=height_2_slopes(Xca, Yca,Z_to_remove_ca,pixel_m,winSz)
Z_to_remove_ca = np.copy(Z_to_remove_ca)
Zx_residual_ca, Zy_residual_ca=height_2_slopes(Xca,Yca,Z_residual_ca,pixel_m,winSz)
Z_residual_ca = np.copy(Z_residual_ca)
show_rifta_height_estimation_result(X_B, Y_B, B, X_P, Y_P, T_P, Xca, Yca,Z_to_remove_ca, Z_residual_ca,Zx_to_remove_ca, Zx_residual_ca,Zy_to_remove_ca, Zy_residual_ca)
plt.show()