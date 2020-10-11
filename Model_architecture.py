# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 12:19:42 2020

@author: Mike
"""

import numpy as np
import pandas as pd
from datetime import datetime

from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Flatten
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, BatchNormalization, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.models import model_from_json
from tensorflow.keras import metrics
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.optimizers import Nadam, Adam

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split as split

import tensorflow.keras.backend as K
import tensorflow.keras.callbacks as Kc




# Disable eager execution
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# Mixed precision
#tf.keras.mixed_precision.experimental.set_policy('float16')
#from tensorflow.keras.mixed_precision import experimental as mixed_precision
#mixed_precision.set_policy('mixed_float16')



import time
from datetime import datetime

from deap import base, creator, tools, algorithms
from bitstring import BitArray

from random import randrange

from PPM_AUTO_EVAL.Eval_helpers import*
from PPM_AUTO_EVAL.HPO_searchspace import *


def GenModel(data_objects, model_params):
    
    # Load the data to check dimensionality
    x_train, y_train = data_objects["x_train"], data_objects["y_train"]
    print("Input data shape:",x_train.shape)
    
    # Model-controlled parameters:
    F_modeltype = data_objects["F_modeltype"]
    
    #######################################################################
    
    # Number of block layers
    BLOCK_LAYERS = model_params["BLOCK_LAYERS"]
    
    
    # Alternative block type setup:
    BLOCK1_TYPE = model_params["BLOCK1_TYPE"]
    BLOCK2_TYPE = model_params["BLOCK2_TYPE"]
    BLOCK3_TYPE = model_params["BLOCK3_TYPE"]
    BLOCK4_TYPE = model_params["BLOCK4_TYPE"]
    
    FC_BLOCK1 = model_params["FC_BLOCK1"]
    FC_BLOCK2 = model_params["FC_BLOCK2"]
    FC_BLOCK3 = model_params["FC_BLOCK3"]
    FC_BLOCK4 = model_params["FC_BLOCK4"]
    
    #CNN related params
    DROPOUT_RATE = model_params["DROPOUT_RATE"]
    
    #FULLY_CONNECTED = model_params["FULLY_CONNECTED"]
    
    NUM_FILTERS = 0#model_params["NUM_FILTERS"]
    KERNEL_SIZE = 0#model_params["KERNEL_SIZE"]
    KERNEL_STRIDE = 0#model_params["KERNEL_STRIDE"]
    POOL_STRIDE = 0#model_params["POOL_STRIDE"]
    POOL_SIZE = 0#model_params["POOL_SIZE"]
    
    PADDING = 0 #used for analysis only
    
    input_dim = (x_train.shape[1], x_train.shape[2])
    
    # Print the configuration to be trained:
        
    print("Hyper params:")
    print("================================"*3)
    print("================================"*3)    
    print("\nArchitecture::")
    print("Blocks:         ",model_params["BLOCK_LAYERS"])
    print("")
    print("Block types:    ",model_params["BLOCK1_TYPE"],model_params["BLOCK2_TYPE"],model_params["BLOCK3_TYPE"],model_params["BLOCK4_TYPE"])
    print("Hidden units:   ", model_params["FC_BLOCK1"], model_params["FC_BLOCK2"], model_params["FC_BLOCK3"], model_params["FC_BLOCK4"])
    print("Dropout:        ",(model_params["DROPOUT_RATE"]))
    print("\nTraining:")
    print("- batch_size:   ",(model_params["batch_size"]))
    print("- optimizer:    ",(model_params["optimizer"]))
    print("- learningrate: ",(model_params["learningrate"]))
    
    print("================================"*3)
    print("================================"*3)
        
    
    #######################################################################
    
    #################### Config analysis #####################
    
    #######################################################################
    """
    ##### Helper functions:
    def Get_outdim_conv(Inputdim, paddingdim, kerneldim, stridedim, filters):
        out_dim = (int((Inputdim + 2*paddingdim - kerneldim)/stridedim +1), filters)
        print("Conv:",out_dim)
        return out_dim
    
    def Get_outdim_maxpool(Inputdim, pooldim, stridedim, filters):
        out_dim = (int(((Inputdim-pooldim)/stridedim +1)), filters)
        print("Maxpool:",out_dim)
        return out_dim
    
    ##### Pre-model building Analysis/validation
    if F_modeltype == "CNN":
        
        # List of blocks that can be added to the model
        blocks_available = []
        print("Pre-build analysis:")
        for block in range(1,BLOCK_LAYERS+1):
            print("-----------")
            if block == 1:
                newdims = Get_outdim_conv(Inputdim=input_dim[0], paddingdim=PADDING, kerneldim=KERNEL_SIZE, stridedim=KERNEL_STRIDE, filters=NUM_FILTERS)
                newdims = Get_outdim_maxpool(Inputdim=newdims[0], pooldim=POOL_SIZE, stridedim=POOL_STRIDE, filters=NUM_FILTERS)
            
            if block > 1:
                newdims = Get_outdim_conv(Inputdim=newdims[0], paddingdim=PADDING, kerneldim=KERNEL_SIZE, stridedim=KERNEL_STRIDE, filters=NUM_FILTERS)
                newdims = Get_outdim_maxpool(Inputdim=newdims[0], pooldim=POOL_SIZE, stridedim=POOL_STRIDE, filters=NUM_FILTERS)
             
            if newdims[0] < 1:
                print("Block ",block," fail")
                if block == 1:
                    print("First block fail - Need different configuration/bounds")
                blocks_available.append(0)
            
            if newdims[0] > 0:
                print("Block",block,"pass: ",newdims)
                blocks_available.append(block)
    """
    #######################################################################
    
    def Make_block(BLOCK_TYPE, BLOCK, DROPOUT_RATE, #FULLY_CONNECTED,
                   NUM_FILTERS, KERNEL_SIZE, KERNEL_STRIDE, POOL_STRIDE, POOL_SIZE, PADDING, input_dim, model_type=None, end_seq=None):
        """
        if model_type == "CNN":
        
            # First layer: Include input dim
            if BLOCK_TYPE == 1 & BLOCK == 1:
                model.add(Conv1D(NUM_FILTERS, kernel_size=KERNEL_SIZE, padding='valid', activation='relu', 
                             strides=KERNEL_STRIDE, use_bias=True, input_shape=input_dim))
                model.add(MaxPooling1D(pool_size=POOL_SIZE, strides=POOL_STRIDE))
               
            # No need to specify input dim
            if BLOCK_TYPE == 1 & BLOCK > 1:
                model.add(Conv1D(NUM_FILTERS, kernel_size=KERNEL_SIZE, padding='valid', activation='relu', strides=KERNEL_STRIDE, use_bias=True, input_shape=input_dim))
                model.add(MaxPooling1D(pool_size=POOL_SIZE, strides=POOL_STRIDE))
        
            if BLOCK_TYPE == 2 & BLOCK == 1:
                model.add(Conv1D(NUM_FILTERS, kernel_size=KERNEL_SIZE, padding='valid', activation='relu', 
                             strides=KERNEL_STRIDE, use_bias=True, input_shape=input_dim))
                model.add(MaxPooling1D(pool_size=POOL_SIZE, strides=POOL_STRIDE))
                model.add(BatchNormalization())
                
            if BLOCK_TYPE == 2 & BLOCK > 1:
                model.add(Conv1D(NUM_FILTERS, kernel_size=KERNEL_SIZE, padding='valid', activation='relu', strides=KERNEL_STRIDE, use_bias=True, input_shape=input_dim))
                model.add(MaxPooling1D(pool_size=POOL_SIZE, strides=POOL_STRIDE))
                model.add(BatchNormalization())
                
            if BLOCK_TYPE == 3 & BLOCK == 1:
                model.add(Conv1D(NUM_FILTERS, kernel_size=KERNEL_SIZE, padding='valid', activation='relu', 
                             strides=KERNEL_STRIDE, use_bias=True, input_shape=input_dim))
                model.add(MaxPooling1D(pool_size=POOL_SIZE, strides=POOL_STRIDE))
                model.add(Dropout(DROPOUT_RATE))
                
            if BLOCK_TYPE == 3 & BLOCK > 1:
                model.add(Conv1D(NUM_FILTERS, kernel_size=KERNEL_SIZE, padding='valid', activation='relu', strides=KERNEL_STRIDE, use_bias=True, input_shape=input_dim))
                model.add(MaxPooling1D(pool_size=POOL_SIZE, strides=POOL_STRIDE))
                model.add(Dropout(DROPOUT_RATE))
            
            if BLOCK_TYPE == 4 & BLOCK == 1:
                model.add(Conv1D(NUM_FILTERS, kernel_size=KERNEL_SIZE, padding='valid', activation='relu', 
                             strides=KERNEL_STRIDE, use_bias=True, input_shape=input_dim))
                model.add(MaxPooling1D(pool_size=POOL_SIZE, strides=POOL_STRIDE))
                model.add(BatchNormalization())
                model.add(Dropout(DROPOUT_RATE))
                
            if BLOCK_TYPE == 4 & BLOCK > 1:
                model.add(Conv1D(NUM_FILTERS, kernel_size=KERNEL_SIZE, padding='valid', activation='relu', strides=KERNEL_STRIDE, use_bias=True, input_shape=input_dim))
                model.add(MaxPooling1D(pool_size=POOL_SIZE, strides=POOL_STRIDE))
                model.add(BatchNormalization())
                model.add(Dropout(DROPOUT_RATE))
        
        """
        if model_type == "LSTM":
            
            if BLOCK == 1:
               FULLY_CONNECTED = FC_BLOCK1
            if BLOCK == 2:
               FULLY_CONNECTED = FC_BLOCK2
            if BLOCK == 3:
               FULLY_CONNECTED = FC_BLOCK3
            if BLOCK == 4:
               FULLY_CONNECTED = FC_BLOCK4
                        
            # For contiunation of the recurrent sequences
            if end_seq == False:
                print("type",BLOCK_TYPE, "Seq = T")
                
                # First layer: Include input dim
                
                # Basic LSTM with recurrent dropout
                if BLOCK_TYPE == 1:
                    model.add(LSTM(FULLY_CONNECTED,  implementation=2, 
                                         input_shape=input_dim,
                                         recurrent_dropout=DROPOUT_RATE, 
                                         return_sequences=True))
                    
                # LSTM with batchNorm and recurrent dropout
                if BLOCK_TYPE == 2:
                    model.add(LSTM(FULLY_CONNECTED,  implementation=2, 
                                         input_shape=input_dim,
                                         recurrent_dropout=DROPOUT_RATE, 
                                         return_sequences=True))
                    model.add(BatchNormalization())
                   
                # LSTM with no dropout
                if BLOCK_TYPE == 3:
                   model.add(LSTM(FULLY_CONNECTED,  implementation=2, 
                                         input_shape=input_dim,
                                         return_sequences=True))
                
                # LSTM with batchNorm and no dropout
                if BLOCK_TYPE == 4:
                   model.add(LSTM(FULLY_CONNECTED,  implementation=2, 
                                         input_shape=input_dim,
                                         return_sequences=True))
                   model.add(BatchNormalization())
                   
            
            # For ending the sequence
            if end_seq == True:
                print("type",BLOCK_TYPE, "Seq = F")
            
                # First layer: Include input dim
                
                # Basic LSTM with recurrent dropout
                if BLOCK_TYPE == 1:
                    model.add(LSTM(FULLY_CONNECTED,  implementation=2, 
                                         recurrent_dropout=DROPOUT_RATE, 
                                         return_sequences=False))
                    
                # LSTM with batchNorm and recurrent dropout
                if BLOCK_TYPE == 2:
                    model.add(LSTM(FULLY_CONNECTED,  implementation=2, 
                                         recurrent_dropout=DROPOUT_RATE, 
                                         return_sequences=False))
                    model.add(BatchNormalization())
                   
                # LSTM with no dropout
                if BLOCK_TYPE == 3:
                   model.add(LSTM(FULLY_CONNECTED,  implementation=2, 
                                         input_shape=input_dim,
                                         return_sequences=False))
                
                # LSTM with batchNorm and no dropout
                if BLOCK_TYPE == 4:
                   model.add(LSTM(FULLY_CONNECTED,  implementation=2, 
                                         input_shape=input_dim,
                                         return_sequences=False))
                   model.add(BatchNormalization())
                   
        return
    
    #######################################################################
    
    ########## Achitecture build #########################################
    
    #######################################################################

    
    model = Sequential()
    
    if F_modeltype == "LSTM":
        
        #Store number of blocks
        blocks_available = BLOCK_LAYERS
        print(F_modeltype)
        for block in range(0, BLOCK_LAYERS):
            
            block = block +1
            print("Adding block",block, "of",BLOCK_LAYERS)
            
            # Figure out whether to end the sequence or not
            Last_block = False
            
            if block == BLOCK_LAYERS:
                print("last block: ",block,"=",BLOCK_LAYERS)
                Last_block = True
                
            
            if block ==1:
                Make_block(BLOCK1_TYPE, block, DROPOUT_RATE, #FULLY_CONNECTED,
                       NUM_FILTERS, KERNEL_SIZE, KERNEL_STRIDE, POOL_STRIDE, POOL_SIZE, PADDING, input_dim, model_type=F_modeltype, end_seq=Last_block)
            if block ==2:
                Make_block(BLOCK1_TYPE, block, DROPOUT_RATE, #FULLY_CONNECTED,
                       NUM_FILTERS, KERNEL_SIZE, KERNEL_STRIDE, POOL_STRIDE, POOL_SIZE, PADDING, input_dim, model_type=F_modeltype, end_seq=Last_block)
            if block ==3:
                Make_block(BLOCK3_TYPE, block, DROPOUT_RATE, #FULLY_CONNECTED,
                       NUM_FILTERS, KERNEL_SIZE, KERNEL_STRIDE, POOL_STRIDE, POOL_SIZE, PADDING, input_dim, model_type=F_modeltype, end_seq=Last_block)
            if block ==4:
                Make_block(BLOCK4_TYPE, block, DROPOUT_RATE, #FULLY_CONNECTED,
                       NUM_FILTERS, KERNEL_SIZE, KERNEL_STRIDE, POOL_STRIDE, POOL_SIZE, PADDING, input_dim, model_type=F_modeltype, end_seq=Last_block)
           
            print("Done.")
                
            model.add(Dense(1)) #, dtype='float32' #Only the softmax is adviced to be float32 
        
    #print(model.summary())
    if BLOCK_LAYERS > 1:
        print("Total number of params:",model.count_params())
    #################### TESTING #########################
    
    """
    # Print model overview
    print(model.summary())
    
    
    
    #compiling the model, creating the callbacks
    model.compile(loss='mae', 
          optimizer='Nadam',
          metrics=['mae'])
    
    
    trainhist = model.fit(x_train, 
                       y_train, 
                       validation_split=0.2,
                       epochs=10, 
                       batch_size=(batch_size))
    
    scores = model.evaluate(x_test, y_test, verbose=1, batch_size=512)
    
    mae_test = scores[1]#/(24.0*3600)
    
    mae_test = mae_test/(24.0*3600)
    mae_test
    """
    
    return model, blocks_available


