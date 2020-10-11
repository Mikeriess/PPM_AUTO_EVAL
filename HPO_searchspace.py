# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 16:53:38 2020
@author: Mike
"""

#### Model settings

"""
Bits:
- Three have 8 possible values (0-7)
- Four have 16 possible values (0-15)
- Five have 32 possible values
- Six have 64 possible values
- Seven have 128 possible values (0-127)
- 8
"""
from bitstring import BitArray
import time

import numpy as np
import pandas as pd
from random import randrange

"""
    hidden_units = 200 #model_params["hidden_units"]
    
    #######################################################################
    
    FULLY_CONNECTED = 200
    
    # Number of block layers
    block_layers = 6 #model_params["block_layers"]
    
    
    # Alternative block type setup:
    BLOCK1_TYPE = randrange(1,4+1)
    BLOCK2_TYPE = randrange(1,4+1)
    BLOCK3_TYPE = randrange(1,4+1)
    BLOCK4_TYPE = randrange(1,4+1)
    BLOCK5_TYPE = randrange(1,4+1)
    BLOCK6_TYPE = randrange(1,4+1)
    
    #CNN related params
    DROPOUT_RATE = 0.5 #model_params["normal_dropout_val"]
    
    NUM_FILTERS = 50 #model_params["twod_filter"]
    KERNEL_SIZE = 6 #model_params["kernel_size"]
    KERNEL_STRIDE = 1 #model_params["strides"]

    POOL_STRIDE = 1
    POOL_SIZE = 3
    PADDING = 0
"""

# Model related hyper parameters

HP1_1_bits = 2 # FC_BLOCK1 (16, 32, 64, 128, 256, 512, 1024, 2048)
HP1_2_bits = 2 # FC_BLOCK2 (16, 32, 64, 128, 256, 512, 1024, 2048)
HP1_3_bits = 2 # FC_BLOCK3 (16, 32, 64, 128)
HP1_4_bits = 2 # FC_BLOCK4 (16, 32, 64, 128)


HP2_bits = 2 # BLOCK_LAYERS (1, 2, 3, 4) 

HP3_bits = 2 # BLOCK1_type (1,2,3,4)
HP4_bits = 2 # BLOCK2_type (1,2,3,4)
HP5_bits = 2 # BLOCK3_type (1,2,3,4)
HP6_bits = 2 # BLOCK4_type (1,2,3,4)

HP7_bits = 2 # DROPOUT_RATE [0, 0.1, 0.25, 0.5]

# Hyper parameters for training (last)

HP8_bits = 2 # batch_size (8, 16, 32, 64, 128, 256, 512, 1024)
HP9_bits = 2 # learningrate (0.0001, 0.001, 0.005, 0.01)
HP10_bits = 1 # optimizer (Adam, Nadam) Radam

##################### REMEMBER - the order of the params matter!

gene_length = (HP1_1_bits + HP1_2_bits + HP1_3_bits + HP1_4_bits) + HP2_bits + HP3_bits + HP4_bits + HP5_bits + HP6_bits + HP7_bits + HP8_bits + HP9_bits + HP10_bits


def GeneConverter(individual):
     
    #Categorical
    FC_BLOCK1 = BitArray(individual[0:HP1_1_bits-1]).uint
    FC_BLOCK2 = BitArray(individual[HP1_1_bits:HP1_1_bits+HP1_2_bits]).uint
    FC_BLOCK3 = BitArray(individual[HP1_2_bits:HP1_2_bits+HP1_3_bits]).uint
    FC_BLOCK4 = BitArray(individual[HP1_3_bits:HP1_3_bits+HP1_4_bits]).uint
    
    #Categorical
    BLOCK_LAYERS = BitArray(individual[HP1_4_bits:HP1_4_bits+HP2_bits]).uint
    
    #Categorical
    BLOCK1_TYPE = BitArray(individual[HP2_bits:HP2_bits+HP3_bits]).uint
    BLOCK2_TYPE = BitArray(individual[HP3_bits:HP3_bits+HP4_bits]).uint
    BLOCK3_TYPE = BitArray(individual[HP4_bits:HP4_bits+HP5_bits]).uint
    BLOCK4_TYPE = BitArray(individual[HP5_bits:HP5_bits+HP6_bits]).uint
    
    #Categorical
    DROPOUT_RATE = BitArray(individual[HP6_bits:HP6_bits+HP7_bits]).uint
    
    #Categorical
    batch_size = BitArray(individual[HP7_bits:HP7_bits+HP8_bits]).uint
    learningrate = BitArray(individual[HP8_bits:HP8_bits+HP9_bits]).uint
    optimizer = BitArray(individual[HP9_bits:HP9_bits+HP10_bits]).uint
    
    #######################
    # Convert the categorical levels
    FC_BLOCK1 = [25, 50, 100, 150][FC_BLOCK1] #, 256, 512, 1024, 2048
    FC_BLOCK2 = [25, 50, 100, 150][FC_BLOCK2]
    FC_BLOCK3 = [25, 50, 100, 150][FC_BLOCK3]
    FC_BLOCK4 = [25, 50, 100, 150][FC_BLOCK4]
    
    # Block layers: less than 7
    BLOCK_LAYERS = [1, 2, 3, 4][BLOCK_LAYERS] #, 5, 6, 7, 8
    
    BLOCK1_TYPE = [1, 2, 3, 4][BLOCK1_TYPE]
    BLOCK2_TYPE = [1, 2, 3, 4][BLOCK2_TYPE]
    BLOCK3_TYPE = [1, 2, 3, 4][BLOCK3_TYPE]
    BLOCK4_TYPE = [1, 2, 3, 4][BLOCK4_TYPE]
    
    DROPOUT_RATE = [0.1, 0.2, 0.3, 0.4][DROPOUT_RATE]
    
    batch_size = [128, 256, 512, 1024][batch_size] # 64, 128, 256, 512 #[16, 32, 64, 128, 256, 512, 1024, 2048]
    learningrate = [0.0001, 0.001, 0.005, 0.01][learningrate]
    optimizer = ["Adam", "Nadam"][optimizer]
        
    modelparams = {"FC_BLOCK1":FC_BLOCK1,
                   "FC_BLOCK2":FC_BLOCK2,
                   "FC_BLOCK3":FC_BLOCK3,
                   "FC_BLOCK4":FC_BLOCK4,
                   
                   "BLOCK_LAYERS":BLOCK_LAYERS,
                   
                   "BLOCK1_TYPE":BLOCK1_TYPE,
                   "BLOCK2_TYPE":BLOCK2_TYPE,
                   "BLOCK3_TYPE":BLOCK3_TYPE,
                   "BLOCK4_TYPE":BLOCK4_TYPE,
                   
                   "DROPOUT_RATE":DROPOUT_RATE,
                   
                   "batch_size":batch_size,
                   "learningrate":learningrate,
                   "optimizer":optimizer,
                   "individual":BitArray(individual).uint}
    
    return modelparams

#TEST: Generate a random individual
#individual = np.random.randint(2, size=(gene_length,))
#GeneConverter(individual)
