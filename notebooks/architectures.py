#!/usr/bin/env python
# coding: utf-8
CPU_COUNT = 4

import os
import sys
import datetime
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import gc
import itertools
from shutil import copyfile
# from contextlib import redirect_stdout
sys.path.append('..')


# In[5]:


from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, Input
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Conv2D, MaxPooling3D, Conv3D, MaxPooling2D, Convolution1D, Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam, RMSprop


# In[6]:


from sklearn.metrics import confusion_matrix


# In[8]:


# setup paths
# pwd = os.getcwd().replace("deepvideoclassification","")
pwd = os.getcwd().replace("notebooks","")
path_cache = pwd + 'cache/'
path_data = pwd + 'data/'

# In[9]:


# In[10]:


from deepvideoclassification.data import Data

# load preprocessing functions
from deepvideoclassification.pretrained_CNNs import load_pretrained_model, load_pretrained_model_preprocessor, precompute_CNN_features
# load preprocessing constants
from deepvideoclassification.pretrained_CNNs import pretrained_model_len_features, pretrained_model_sizes


# # Confusion Matrix

# In[20]:


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# # Architecture class (contains keras model object and train/evaluate method, writes training results to /models/)

# In[21]:


class Architecture(object):
    
    def __init__(self, model_id, architecture, sequence_length, 
                frame_size = None, 
                pretrained_model_name = None, pooling = None,
                sequence_model = None, sequence_model_layers = None,
                layer_1_size = 0, layer_2_size = 0, layer_3_size = 0, 
                dropout = 0, convolution_kernel_size = 3, 
                model_weights_path = None, 
                batch_size = 32, 
                verbose = False,
                logger = None):
        """
        Model object constructor. Contains Keras model object and training/evaluation methods. Writes model results to /models/_id_ folder
        
        Architecture can be one of: 
        
        :model_id: integer identifier for this model e.g. 1337
        
        :architecture: architecture of model in [image_MLP_frozen, image_MLP_trainable, video_MLP_concat, video_LRCNN_frozen, video_LRCNN_trainable, C3D, C3Dsmall]
        
        :sequence_length: number of frames in sequence to be returned by Data object
        
        :frame_size: size that frames are resized to (different models / architectures accept different input sizes - will be inferred if pretrained_model_name is given since they have fixed sizes)
        :pretrained_model_name: name of pretrained model (or None if not using pretrained model e.g. for 3D-CNN)
        :pooling: name of pooling variant (or None if not using pretrained model e.g. for 3D-CNN or if fitting more non-dense layers on top of pretrained model base)
        
        :sequence_model: sequence model in [LSTM, SimpleRNN, GRU, Convolution1D]
        :sequence_model_layers: default to 1, can be stacked 2 or 3 (but less than 4) layer sequence model (assume always stacking the same sequence model, not mixing LSTM and GRU, for example)
        
        :layer_1_size: number of neurons in layer 1
        :layer_2_size: number of neurons in layer 2
        :layer_3_size: number of neurons in layer 3
        :layer_4_size: number of neurons in layer 4 
        
        
   
    
    def make_last_layers_trainable(self, num_layers):
        """
        Set the last *num_layers* non-trainable layers to trainable  

        NB to be used with model_base and assumes name = "top_xxx" added to each top layer to know 
        to ignore that layer when looping through layers from top backwards

        :num_layers: number of layers from end of model (that are currently not trainable) to be set as trainable
        """

        # get index of last non-trainable layer
        # (the layers we added on top of model_base are already trainable=True)
        # ...
        # need to find last layer of base model and set that (and previous num_layers)
        # to trainable=True via this method

        # find last non-trainable layer index
        idx_not_trainable = 0
        for i, l in enumerate(self.model.layers):
            if "top" not in l.name:
                idx_not_trainable = i

        # set last non-trainable layer and num_layers prior to trainable=True
        for i in reversed(range(idx_not_trainable-num_layers+1, idx_not_trainable+1)):
            self.model.layers[i].trainable = True
        
        if self.verbose:
            self.logger.info("last {} layers of CNN set to trainable".format(num_layers))
            

    def fit(self, fit_round, learning_rate, epochs, patience):
        """
        Compile and fit model for *epochs* rounds of training, dividing learning rate by 10 after each round

        Fitting will stop if val_acc does not improve for at least patience epochs

        Only the best weights will be kept

        The model is saved to /models/*model_id*/

        Good practice is to decrease the learning rate by a factor of 10 after each plateau and train some more 
        (after first re-loading best weights from previous training round)...

        for example (not exact example because this fit method has been refactored into the architecture object but the principle remains):
            fit_history = fit(model_id, model, data, learning_rate = 0.001, epochs = 30)
            model.load_weights(path_model + "model.h5")
            model = fit(model, 5)
            fit_history = train(model_id, model, data, learning_rate = 0.0001, epochs = 30)

        :fit_round: keep track of which round of learning rate annealing we're on
        :learning_rate: learning rate parameter for Adam optimizer (default is 0.001)
        :epochs: number of training epochs per fit round, subject to patience setting - good default is 30 or more
        :patience: how many epochs without val_acc improvement before stopping fit round (good default is 5) 
        
        :verbose: print progress

        """

        # create optimizer with given learning rate 
        opt = Adam(lr = learning_rate)

        # compile model
        self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        # setup training callbacks
        callback_stopper = EarlyStopping(monitor='val_acc', patience=patience, verbose=self.verbose)
        callback_csvlogger = CSVLogger(self.path_model + 'training_round_' + str(fit_round) + '.log')
        callback_checkpointer = ModelCheckpoint(self.path_model + 'model_round_' + str(fit_round) + '.h5', monitor='val_acc', save_best_only=True, verbose=self.verbose)
        callbacks = [callback_stopper, callback_checkpointer, callback_csvlogger]

        # fit model
        if self.data.return_generator == True:
            # train using generator
            history = self.model.fit_generator(generator=self.data.generator_train,
                validation_data=self.data.generator_valid,
                use_multiprocessing=True,
                workers=CPU_COUNT,
                epochs=epochs,
                callbacks=callbacks,
                shuffle=True,
                verbose=True)
        else:
            # train using full dataset
            history = self.model.fit(self.data.x_train, self.data.y_train, 
                validation_data=(self.data.x_valid, self.data.y_valid),
                batch_size=self.batch_size,
                epochs=epochs,
                callbacks=callbacks,
                shuffle=True,
                verbose=False)

        # get number of epochs actually trained (might have early stopped)
        epochs_trained = callback_stopper.stopped_epoch
        
        if epochs_trained == 0:
            # trained but didn't stop early
            if len(history.history) > 0:
                epochs_trained = (epochs - 1)
        else:
            # best validation accuracy is (patience-1) epochs before stopped
            epochs_trained -= (patience - 1)
            
           
        
        # return fit history and the epoch that the early stopper completed on
        return history, epochs_trained

    
    def train_model(self, epochs = 20, patience = 3):
        """
        Run several rounds of fitting to train model, reducing learning rate after each round
        
        Progress and model parameters will be saved to the model's path e.g. /models/1/
        
        """
        
        # init results with architecture params
        results = self.__dict__.copy()
        results['data_total_rows_train'] = self.data.total_rows_train
        results['data_total_rows_valid'] = self.data.total_rows_valid
        results['data_total_rows_test'] = self.data.total_rows_test
        # delete non-serializable objects from the architecture class
        del results['model']
        del results['data']
        del results['logger']
        results['model_param_count'] = self.model.count_params()
        
        
        ###############
        ### Train model
        ###############
        
        # start training timer
        start = datetime.datetime.now()
        results['fit_dt_train_start'] = start.strftime("%Y-%m-%d %H:%M:%S")
        
        
        