import keras.backend as K
from keras import callbacks
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, Activation, MaxPooling1D, Conv1D, Input, add, BatchNormalization, AveragePooling1D, LeakyReLU
from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, Callback, LearningRateScheduler
from keras.utils import multi_gpu_model, Sequence, to_categorical
from keras.losses import binary_crossentropy,categorical_crossentropy
from keras.regularizers import l2
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
import os
import json
from collections import Counter
from datetime import date
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from math import pow, floor
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from tqdm import tqdm_notebook
from keras_tqdm import TQDMNotebookCallback
from keras_radam import RAdam
from keras.constraints import maxnorm
from sklearn.utils import class_weight

def model_build(params):
    def conv_block(input_data, n_filters, filter_size, index):
        x = Conv1D(n_filters, filter_size, strides=2, padding='same', 
                   name='conv_block' + str(index) + '_' + 'conv_1', 
                   kernel_initializer=params["conv_init"])(input_data)
        x = BatchNormalization(name='conv_block' + str(index) + '_' + 'BN_1')(x)
        x = Activation('relu', name='conv_block' + str(index) + '_' + 'relu_1')(x)
        x = Conv1D(n_filters, filter_size, strides=1, padding='same', 
                   name='conv_block' + str(index) + '_' + 'conv_2',
                   kernel_initializer=params["conv_init"])(x)
        x = BatchNormalization(name='conv_block' + str(index) + '_' + 'BN_2')(x)
        x = Activation('relu', name='conv_block' + str(index) + '_' + 'relu_2')(x)


        shortcut = Conv1D(n_filters, filter_size, strides=2, padding='same',
                         name='conv_block' + str(index) + '_' + 'shortcut_conv', 
                          kernel_initializer=params["conv_init"])(input_data)
        shortcut = BatchNormalization(name='conv_block' + str(index) + '_' + 'shortcut_BN')(shortcut)
        x = add([x, shortcut], name='conv_block' + str(index) + '_' + 'add')
        x = Activation('relu', name='conv_block' + str(index) + '_' + 'relu_3')(x)
        
        return x

    def identity_block(input_data, n_filters, filter_size, index):
        x = Conv1D(n_filters, filter_size, strides=1, padding='same', 
                   name='identity_block' + str(index) + '_' + 'conv_1', 
                   kernel_initializer=params["conv_init"])(input_data)
        x = BatchNormalization(name='identity_block' + str(index) + '_' + 'BN_1')(x)
        x = Activation('relu', name='identity_block' + str(index) + '_' + 'relu_1')(x)
        x = add([x, input_data], name='identity_block' + str(index) + '_' + 'add')
        x = Activation('relu', name='identity_block' + str(index) + '_' + 'relu_2')(x)
        
        return x
  
    input_ecg = Input(shape=(5000, 4), name='input')
    x = Conv1D(filters=params["conv_num_filters"][0], kernel_size=15, 
               strides=2, padding='same', kernel_initializer=params["conv_init"], name='conv_2')(input_ecg)
    x = BatchNormalization(name='BN_2')(x)
    x = Activation('relu', name='relu_2')(x)
    x = MaxPooling1D(name='max_pooling_1')(x)
    

    for i in range(2):
        x = conv_block(x, n_filters=params["conv_num_filters"][i], filter_size=params["conv_filter_size"], index=i + 1)
        x = MaxPooling1D(name='max_pooling_' + str(i + 2))(x)
        x = identity_block(x, n_filters=params["conv_num_filters"][i], 
                               filter_size=params["conv_filter_size"], index=i + 1)
            
    x = AveragePooling1D(name='average_pooling')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(params["dense_neurons"], kernel_regularizer=l2(params["l2"]), name='FC1')(x)
    x = Activation('relu', name='relu_3')(x)
    x = Dropout(rate=params["dropout"])(x)
    x = Dense(params["dense_neurons"], kernel_regularizer=l2(params["l2"]), name='FC2')(x)
    x = Activation('relu', name='relu_4')(x)
    x = Dropout(rate=params["dropout"])(x)
    x = Dense(params["disease_num"], activation='sigmoid', name='output')(x)

    model = Model(inputs=input_ecg, outputs=x)

    
    return model

def multilabel_loss(y_true, y_pred):
        return K.sum(binary_crossentropy(y_true, y_pred))

def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * pow(drop,floor((1+epoch)/epochs_drop))
    return lrate
def model_train(model,train_id, train_label, val_id, val_label, params):
    
    
    class DataGenerator(Sequence):
        """
        Generate data for fit_generator.
        """
        def __init__(self, data_ids, labels, batch_size, n_classes, shuffle=True):
            self.data_ids = data_ids
            self.labels = labels
            self.batch_size = batch_size
            self.n_classes = n_classes
            self.shuffle = shuffle
            self.on_epoch_end()

        def __len__(self):
            """
            Denote the number of batches per epoch.
            """
            return int(len(self.data_ids) / self.batch_size)

        def __getitem__(self, index):
            """
            Generate one batch of data.
            """
            # Generate indexes of the batch
            indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

            # Find list of IDs
            data_ids_temp = [self.data_ids[k] for k in indexes]

            # Generate data
            X, y = self.__data_generation(data_ids_temp)

            return X, y

        def on_epoch_end(self):
            """
            Update indexes after each epoch.
            """
            self.indexes = np.arange(len(self.data_ids))
            if self.shuffle == True:
                np.random.shuffle(self.indexes)

        def __data_generation(self, data_ids_temp):
            """
            Generate data containing batch_size samples.
            """
            # Generate data
            X = np.empty((self.batch_size, 5000, 12))
            y = np.empty((self.batch_size, self.n_classes), dtype=int)
            for i, ID in enumerate(data_ids_temp):
                X[i] = np.load(ID)
                y[i] = self.labels[ID]

            return X, y


    #momentum = 0.9
    #decay_rate = 0.0
    #sgd = SGD(lr=params["learning_rate"], momentum=momentum, decay=decay_rate, nesterov=False)
    #model.compile(loss=multilabel_loss, optimizer=sgd, metrics=['accuracy']) 
    #lrate = LearningRateScheduler(step_decay)
    
    label_binarizer_ = LabelBinarizer().fit(train_label)
    y_ = label_binarizer_.transform(train_label)
    class_weigh = class_weight.compute_class_weight('balanced',
                                                np.unique(train_label),
                                                 train_label)
    cw = dict(enumerate(class_weigh))
    model.compile(loss=multilabel_loss, optimizer=RAdam(lr=params["learning_rate"]), metrics=['accuracy'])      

    my_callbacks = [EarlyStopping(monitor='val_loss', patience=8, verbose=2),#monitor: 需要监视的量，val_loss，val_acc,patience: 当early stop被激活(如发现loss相比上一个epoch训练没有下降)，则经过patience个epoch后停止训练
                    ReduceLROnPlateau(monitor='val_loss', factor=0.06, patience=4, 
                                      min_lr=0.00000001, verbose=1),#  允许基于一些验证测量对学习率进行动态的下降,factor (float) – 学习率衰减的乘数因子。patience (int) – 没有改善的迭代epoch数量，这之后学习率会降低。
                    TQDMNotebookCallback(leave_inner=True, leave_outer=True)]#进度条
#                     TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)]

    model.fit(
    [train_id], y_,batch_size=params["batch_size"], epochs=200, verbose=2,
    validation_data=([val_id, label_binarizer_.transform(val_label)]),
    callbacks=my_callbacks)
    Y_test = np.argmax(label_binarizer_.transform(val_label), axis=1)
    predict = model.predict([val_id])#这里假设你的GT标注也是整数 [1,2,10,1,2,1]
    y_pred=np.argmax(predict,axis=1)

    print(classification_report(Y_test, y_pred))

    return model

def lr_scheduler(epoch):
    rate, step = 0.1, 800
    power = epoch // step
    return 0.001 * (rate) ** power

def model_save(model):
    today = date.today()
    # Save the model
    model.save('multilabel_model_' + today.strftime("%m%d") + '.h5')
    # # Save the weights as well
    model.save_weights('multilabel_model_weights_' + today.strftime("%m%d") + '.h5')

def model_load(h5_name):
    # This code can load the whole model
    model = load_model(h5_name)
    # If necesssary, you can create a new model using the weights you have got.
    # Fisrt create a new model...
    # Then load the weights
    # model.load_weights('model_weights_0805.h5')
    return model

def model_eval(model, params):
    def plot_roc(name, labels, predict_prob, cur_clr):
        fp_rate, tp_rate, thresholds = roc_curve(labels, predict_prob)
        roc_auc = auc(fp_rate, tp_rate)
        plt.title('ROC')
        plt.plot(fp_rate, tp_rate, cur_clr, label= name + "'s AUC = %0.4f" % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        
    def plot_confusion_matrix(name, cm, title='Confusion Matrix', cmap='Blues'):
        labels = ['Non-' + name, name]
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(len(labels)))
        plt.xticks(xlocations, labels, rotation=30)
        plt.yticks(xlocations, labels)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    
        cm_normalized = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        ind_array = np.arange(len(labels))
        x, y = np.meshgrid(ind_array, ind_array)
    
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = cm_normalized[y_val][x_val]
            plt.text(x_val, y_val, "%0.4f" %(c,), color='black', fontsize=15, va='center', ha='center')
    
    # Visualize the classification result
    # First load the test set into memory
    X_test = []
    y_test = []
    for i in test_id:
        X_test.append(np.load(i))
    for i in range(len(test_id)):
        y_test.append(test_label[test_id[i]])

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    test_pos_predict = model.predict(X_test)
    test_predict_onehot = (test_pos_predict >= 0.5).astype(int)

    abbr_list = params["abbr_list"]

    today = date.today()
    # ROC & AUC
    plt.figure(figsize=(24, 20))
    for i in range(len(abbr_list)):
        plt.subplot(5, 6, i+1)
        plot_roc(abbr_list[i], y_test[:, i], test_pos_predict[:, i], 'blue')

    plt.tight_layout()
    plt.savefig('multilabel_roc_' + today.strftime("%m%d") + '.png')
    
    # Confusion matrix
    conf_matrix = []
    for i in range(len(abbr_list)):
        conf_matrix.append(confusion_matrix(y_test[:, i], test_predict_onehot[:, i]))
    plt.figure(figsize=(42, 35))
    for i in range(len(abbr_list)):
        plt.subplot(5, 6, i+1)
        plot_confusion_matrix(abbr_list[i], conf_matrix[i])

    plt.tight_layout()
    plt.savefig('multilabel_conf_' + today.strftime("%m%d") + '.png')
    
def plot_roc(name, labels, predict_prob, cur_clr):
    fp_rate, tp_rate, thresholds = roc_curve(labels, predict_prob)
    roc_auc = auc(fp_rate, tp_rate)
    plt.title('ROC')
    plt.plot(fp_rate, tp_rate, cur_clr, label= name + "'s AUC = %0.4f" % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    
def plot_confusion_matrix(name, cm, title='', cmap='Blues'):
    labels = ['Non-' + name, name]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=25)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=30, fontsize=25)
    plt.yticks(xlocations, labels, rotation=30, fontsize=25)
    plt.ylabel('Committee consensus label', fontsize=25)
    plt.xlabel('Model predicted label', fontsize=25)
    
    cm_normalized = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        plt.text(x_val, y_val, "%0.3f" %(c,), color='black', fontsize=25, va='center', ha='center')