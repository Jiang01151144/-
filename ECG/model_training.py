import os
import collections
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from tqdm import tqdm_notebook
from keras_tqdm import TQDMNotebookCallback
from keras_radam import RAdam
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,classification_report
from modelbuild import model_build, model_train, model_save, model_load, plot_roc, plot_confusion_matrix
from ecg_preprocessing import load_and_label, normalize,load_cardiologist_test_set,multiclass_val_split,all_split
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import sem, t
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

import os
import json
from collections import Counter
from datetime import date
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import time
import pywt
import numpy as np

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config)) 

tf.test.is_gpu_available()

# Load parameters
params = json.load(open('config.json', 'r'))

# Load data and label
Classes = ['RVH', 'RVEH', 'RAH', 'RAEH','LVH','LVEH','LAH','LAEH','Normal'] 
Labels = ['0','0','1','1','2','2','3','3','4']
DATA_path='/mnt/data/房室肥大/DatainNPY/'
Numbers = [641, 2031, 170, 186, 62, 11796, 208, 494, 25960]
collection = collections.namedtuple('format', ['data', 'label', 'file'])

data_by_class = { 
    f'{Classes[0]}':[],
    f'{Classes[1]}':[],
    f'{Classes[2]}':[],
    f'{Classes[3]}':[],
    f'{Classes[4]}':[],
    f'{Classes[5]}':[],
    f'{Classes[6]}':[],
    f'{Classes[7]}':[],
    f'{Classes[8]}':[],

}
data_by_class=load_and_label(Classes, Labels,Numbers,data_by_class,collection)
all_data, all_label = [],[]
for idx in range(len(Labels)):
    for item in data_by_class[f'{Classes[idx]}']:
        w = pywt.Wavelet('db5') # 选用Daubechies8小波
        maxlev = pywt.dwt_max_level(len(item.data[:,[0,1,6,10]]), w.dec_len)

        coeffs = pywt.wavedec(item.data[:,[0,1,6,10]], 'db5', level=maxlev)
        cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
        threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
        # 将高频信号cD1、cD2置零
        cD1.fill(0)
        cD2.fill(0)
        # 将其他中低频信号按软阈值公式滤波
        for i in range(1, len(coeffs) - 2):
            coeffs[i] = pywt.threshold(coeffs[i], threshold)
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold)
        rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
        all_data.append(rdata)
        all_label.append(item.label)
        
normalized = [normalize(tmp_data) for tmp_data in all_data]
norm_data = np.array(normalized)
all_label = np.array(all_label)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=250)
trn_idx, test_idx = list(sss.split(norm_data, all_label))[0]

train_data, train_label = norm_data[trn_idx], all_label[trn_idx]
test_data, test_label = norm_data[test_idx], all_label[test_idx]

# build model
model = model_build(params)

model = model_train(model,train_data, train_label, test_data, test_label, params)





    



