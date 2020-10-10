# -*- coding: utf-8 -*-
"""
Created on Fri May  1 19:29:28 2020

@author: Mike
"""

##############################################################################
"""
instanceno = "1"

# Specify which configfile to store:

"""

instanceno = "" #"0" ##### Removed for final model training
configfilename = "configfile"+instanceno+".csv"


##############################################################################

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

import time
from datetime import datetime

from deap import base, creator, tools, algorithms
from bitstring import BitArray




from Eval_helpers import *
from Reporting import *
from Model_architecture import *
from HPO_searchspace import *




# Callback for tracking training time per epoch:
class TimeHistory(Kc.Callback): #callbacks.
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
            
            


def prepare_dataset(suffix=None, sample=1.0):
    """
    This function is supposed to load and prepare the dataset,
    based on the 
    """
    filepath = "data/"

    ## Cope with the numpy picke error:
    np_load_old = np.load # save np.load
    
    # modify the default parameters of np.load
    newload = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

    ## Load the numpy files
    x_train = newload(filepath+suffix+"_X_train.npy")
    x_test = newload(filepath+suffix+"_X_test.npy")
    y_train = newload(filepath+suffix+"_y_train.npy")
    y_test = newload(filepath+suffix+"_y_test.npy")
    y_t_train = newload(filepath+suffix+"_y_t_train.npy")
    y_t_test = newload(filepath+suffix+"_y_t_test.npy")
    
    ## load inference tables for evaluation and analysis
    Inference_train = pd.read_csv("data/"+suffix+"_Inference_train.csv")
    Inference_test = pd.read_csv("data/"+suffix+"_Inference_test.csv")
    
    ## Event-log characteristics
    n = x_train.shape[0]
    maxlen = x_train.shape[1]
    num_features = x_train.shape[2]  

    ## Sub-sampling the data:
    if sample < 1.0:
        
        #Get new size
        newsize = int(np.round(n*sample,decimals=0))
        
        #Subset based on index
        selected_ids = np.random.randint(n, size=newsize)
        x_train, y_train = x_train[selected_ids,:], y_train[selected_ids,:]

    ## Return the main objects as a dictionary
    data_objects = {"x_train":x_train, 
                    "x_test": x_test,
                    "y_train":y_train,
                    "y_test":y_test,
                    "y_t_train":y_t_train,
                    "y_t_test":y_t_test,
                    "maxlen":maxlen,
                    "num_features":num_features,
                    "Inference_train":Inference_train,
                    "Inference_test":Inference_test}
    return data_objects



def train_model(data_objects, model_params, final_model=False):
    # To prevent tensorboard errors
    K.clear_session()
    
    #### Load the data ########
    x_train, y_train = data_objects["x_train"], data_objects["y_train"]
    #x_test, y_test = data_objects["x_test"], data_objects["y_test"]
        
    #################### Solution ##########################
    # Load the settings of the current experiment:
    configfile = pd.read_csv(configfilename)   
    RUN = configfile["RUN"][0]
    epochs = int(configfile["F_lofi_epochs"][0])
    epochs_final = configfile["Finalmodel_epochs"][0]
    model_type = configfile["F_modeltype"][0]
    
    #Save model type to data objects:
    data_objects["F_modeltype"] = model_type

    #################### Reporting #########################
    #file = str(model_params["individual"])
    
    if final_model == False:
        #Log training history of the individual to csv
        csvfilename = "experiments/" + str(RUN)+ "/train_logfiles/" + str(RUN) + "_" + str(model_params["individual"]) +".csv"
        csv_logger = CSVLogger(csvfilename, append=False, separator=',')
    
    if final_model == True:
        #Log training history of the individual to csv
        csvfilename = "experiments/Final_models/train_logfiles/"+"Trainlog_experiment_" + str(RUN) + "_" + str(model_params["individual"]) +".csv"
        csv_logger = CSVLogger(csvfilename, append=False, separator=',')
    
    ##########################################################
    # Model Type/Architecture
    ##########################################################
    
    # Generate the model
    model, blocks_built = GenModel(data_objects, model_params)

    # Conditional multi-GPU training:
    #if final_model == False and epochs > 1:
        #model = multi_gpu_model(model, gpus=2)
    
    #if final_model == True:
        #model = multi_gpu_model(model, gpus=2)
    
    ##########################################################
    # Train assist
    ##########################################################

    time_callback = TimeHistory()
    
    #Early stopping for the initial search
    earlystop_patience = 20#10 #42
    
    early_stopping = EarlyStopping(patience=earlystop_patience)    
    
    # Batch size:
    batch_size = model_params["batch_size"]
    
    # LR
    learningrate = model_params["learningrate"]
    
    # Optimizer
    if model_params["optimizer"] == "Adam":
        optimizer = Adam(learning_rate=learningrate)
    
    if model_params["optimizer"] == "Nadam":
        optimizer = Nadam(learning_rate=learningrate)
       
    ########################################################
    # Initial Model search settings
    ########################################################
    
    if final_model == False:
        
        #Filename for the model: INTIAL
        filename = "experiments/" + str(RUN)+ "/models/" + str(model_params["individual"]) +".h5"
        
        model_checkpoint = ModelCheckpoint(filename, 
                                   monitor='val_loss', 
                                   verbose=0, 
                                   save_best_only=True, 
                                   mode='auto')
        
        #Learning rate for model search
        #earlystop_patience = 
        
        #early_stopping = EarlyStopping(patience=earlystop_patience) # 42 for Navarini paper
        
        #compiling the model, creating the callbacks
        model.compile(loss='mae', #L1 loss
              optimizer=optimizer, 
              metrics=['mean_squared_error', 'mae', 'mape'])
        
        # Store starttime
        start_time = time.time()
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  callbacks=[early_stopping,
                             model_checkpoint,
                            csv_logger, 
                            time_callback],
                  epochs=epochs,
                  verbose=1,
                  validation_split=0.2)
        
        
        # REPLACE with best version of the model (checkpoint)
        if epochs > 1:
            from keras.models import load_model
            model = load_model(filename)
        
    ########################################################
    # Final model settings
    ########################################################
    
    if final_model == True:
        
        #Load the winner model:
        modeldestination = "experiments/" + str(RUN)+ "/models/" + str(model_params["individual"]) +".h5"
        
        from keras.models import load_model
        model = load_model(modeldestination)
        
        
        #Filename for the model: FINAL
        filename = "experiments/Final_models/models/" +"Experiment_"+str(RUN)+"" + str(model_params["individual"]) +".h5"
        
        model_checkpoint = ModelCheckpoint(filename, 
                                   monitor='val_loss', 
                                   verbose=0, 
                                   save_best_only=True, 
                                   mode='auto')
        
        #Learning rate for full  model
        #learningrate = 0.001
        
        earlystop_patience = 20#10 #42
        
        early_stopping = EarlyStopping(patience=earlystop_patience) # 42 for Navarini paper
        
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', 
                                       factor=0.5, 
                                       patience=10, 
                                       verbose=0, 
                                       mode='auto', 
                                       epsilon=0.0001, 
                                       cooldown=0, 
                                       min_lr=0)

        #compiling the model, creating the callbacks
        model.compile(loss='mae', #L1 loss
              optimizer=optimizer, 
              metrics=['mae']) #'mean_squared_error','mae','mape'
        
        
        # Store starttime
        start_time = time.time()
        
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  callbacks=[early_stopping, 
                            model_checkpoint,
                            lr_reducer,
                            csv_logger,
                            time_callback],
                  epochs=epochs_final,
                  verbose=1,
                  validation_split=0.2)
        
        # REPLACE with best version of the model (checkpoint)
        #from keras.models import load_model
        model = load_model(filename)
            
    
    ########################################################
    # Store results
    ########################################################
    
    # Store endtime
    end_time = time.time()

    # Store epoch times in already existing CSV-history
    epoch_times = time_callback.times
    
    hist = pd.read_csv(csvfilename)
    hist["duration_sec"] = epoch_times
    hist.to_csv(csvfilename,index=False)
    
    
    ## Save model for later inference
    model.save(filename)
    print("Done. Model saved as: ", filename)
    
    ## Store model in data object
    data_objects["model"] = model
        
    ## Store the parameters in data object:
    model_params["model_blocks_built"] = blocks_built
    data_objects["model_params"] = model_params

    #Store number of epochs
    data_objects["epochs"] = len(hist)

    ## Log the full time for training:
    Time_sec = end_time - start_time

    ## Store train time in data object
    data_objects["train_time"] = Time_sec
    
    ## Store the individual in data object
    data_objects["individual"] = model_params["individual"]

    return data_objects




def evaluate_model(data_objects, mode="first-off", final_model=False, TS=True):
    eval_TS = TS
    """
    Description
    ----------
    Function for evaluating a model based on all available criteria;
    Accuracy, earliness, temporal stability, UCL-violations detected.
    
    Runned in sequence after training a model, or for evaluating previously
    trained models.
    
    Generates report files if runned in sequence after training.
    
    If used for final_model evaluation, filenames have Final_ as prefix.
    
    Parameters
    ----------
    data_objects : TYPE
        Dictionary with objects needed.
    mode : TYPE, optional
        Evaluate during training, or post-training. 
        The default is "first-off".

    Returns
    -------
    mae_test : TYPE
        float with MAE on test period
    maepe_test : TYPE
        float with MAEPE on test period

    """
    
    # Load the training data
    #x_train, y_train = data_objects["x_train"], data_objects["y_train"]
    x_test, y_test = data_objects["x_test"], data_objects["y_test"]

    # Load inference tables
    Inference_train = data_objects["Inference_train"]
    Inference_test = data_objects["Inference_test"]   

    # Load the model
    model = data_objects["model"]
    
    # Predict on inference table
    Inference_test["y_pred"] = model.predict(x_test, verbose=1, batch_size=2048)
    
    # Time information
    time_taken = 0
            
    now = datetime.now() # current date and time
    
    timestamp = now.strftime("%Y/%m/%d, %H:%M:%S")
    
    if "train_time" in data_objects:
        time_taken = data_objects["train_time"]
    
    model_params = data_objects["model_params"]
    
    epochs = data_objects["epochs"]
    
    """
    Differentiation: only do UCL for final models
    """
    if final_model == True:
        print("Final-model evaluation..")
         # Identify the individual
        individual_numid = str(data_objects["individual"])
        
        # Load config-file
        configfile = pd.read_csv(configfilename)
        
        RUN = configfile["RUN"][0]
        F_dataset = configfile["F_dataset"][0]
        F_modelselection = configfile["F_modelselection"][0]
        F_mutation_prob = configfile["F_mutation_prob"][0]
    
        # Save information about experiment
        Inference_test["RUN"] = [RUN]*len(Inference_test)
        Inference_test["F_dataset"] = [F_dataset]*len(Inference_test)
        Inference_test["F_modelselection"] = [F_modelselection]*len(Inference_test)
        Inference_test["F_mutation_prob"] = [F_mutation_prob]*len(Inference_test)
        
        
        # Calculate earliness    
        Inference_test = Earliness(Inference_test)
        
        # Calculate temporal stability
        Inference_test = TemporalDiff(Inference_test)
        
        ########## ACCURACY #######################
        mae_test = np.mean(Inference_test["MAE"])/(24.0*3600)
        mae_norm_test = np.mean(Inference_test["MAE_norm"])#/(24.0*3600)
        
        ########## EARLINESS ######################
        
        mep_test = np.mean(Inference_test["MEP"])#/(24.0*3600)
        mep_norm_test = np.mean(Inference_test["MEP_norm"])#/(24.0*3600)
        
        maepe_test = np.mean(Inference_test["MAEPE"])
                
        ########### TEMPORAL STABILITY ############
        ts_sign = np.mean(Inference_test["sign_diff"])
        ts_mag = np.mean(Inference_test["sign_magnitude"])
        
        print('Test MAE:        ', mae_test, ' (days)')
        print('Test MEP:        ', mep_test)
        print('Test MAEPE:      ', maepe_test)
        print('Test TS-sign-diff',ts_sign)
        print('Test TS-sign-mag ',ts_mag)
        print("================================"*3)
        ########### UCL VIOLATIONS IN TEST PERIOD ########
        
        # Find UCL in train table (or window)
        UCL = find_UCL(Inference_train)
        
        # Mark violations
        Inference_test = UCL_eval(Inference_test, UCL)
        
        # Get fitness stats: CASE LEVEL
        precision, recall, accuracy, f1, Inference_test = UCL_eval_class(Inference_test, level="case")
        
        # Get fitness stats: EVENT LEVEL
        precision_ev, recall_ev, accuracy_ev, f1_ev, Inference_test = UCL_eval_class(Inference_test, level="event")

        # Dump the inference tables for debugging
        Inference_test.to_csv("experiments/Final_models/inference_tables/"+"Final_"+"Inf_test_"+str(RUN)+".csv",index=False)
        

        
        #### Differentiate the reports: Final model has all info
        
        # Generate report with parameters of interest
        results = {"individual_numid":individual_numid,
                               "Time":timestamp,
                               "Traintime":time_taken, 
                               "Epochs":epochs,
                               "MAE":mae_test,
                               "MEP":mep_test,
                               "MAEPE":maepe_test,
                               "TS_sign_diff":ts_sign,
                               "TS_sign_mag":ts_mag,
                               "UCL_precision":precision,
                               "UCL_recall":recall,
                               "UCL_accuracy":accuracy,
                               "UCL_f1":f1,
                               "EV_UCL_precision":precision_ev,
                               "EV_UCL_recall":recall_ev,
                               "EV_UCL_accuracy":accuracy_ev,
                               "EV_UCL_f1":f1_ev}
                
        report = dict(results, **model_params)
        report = pd.DataFrame(report, index=[0])
 
        # Adding "final_" + search type to file name for final model
        report.to_csv("experiments/Final_models/individuals/"+"Final_" + str(RUN) + ".csv",index=False) 
            
        output = report


    if final_model == False:
        print("Initial-model evaluation..")
        # Calculate earliness    
        Inference_test = Earliness_and_TS(Inference_test, EAR=True, TS=eval_TS)
        
        ########## ACCURACY #######################
        # Evaluate on the test period
        #scores = model.evaluate(x_test, y_test, verbose=1, batch_size=512)
        #mae_test = scores[2]/(24.0*3600)
        
        mae_test = np.mean(Inference_test["MAE"])/(24.0*3600)
        mae_norm_test = np.mean(Inference_test["MAE_norm"])#/(24.0*3600)
        
        ########## EARLINESS ######################
        
        mep_test = np.mean(Inference_test["MEP"])#/(24.0*3600)
        mep_norm_test = np.mean(Inference_test["MEP_norm"])#/(24.0*3600)
        
        maepe_test = np.mean(Inference_test["MAEPE"])
        ########### TEMPORAL STABILITY ############
        if eval_TS == True:
            # Proportion of sign changes
            ts_sign = np.mean(Inference_test["SSC"])
            # Sum of magnitude of sign changes
            ts_mag = np.mean(Inference_test["SMSC"])
            
        #Check for NaN: If nan, penalize to prevent crashing the GA
        if np.isnan(mae_test):
            mae_test = 9999
            maepe_test = 9999
            if eval_TS == True:
                ts_sign = 9999
                ts_mag = 9999
        print('_'*60)
        print('Test MAE:     ', mae_test, ' (days)')
        print('Test MEP:     ', mep_test)
        print('Test MAEPE:   ', maepe_test, '')
        if eval_TS == True:
            print('TS:')
            print('Test TS-sign-diff',ts_sign, "(proportion sign changes)")
            print('Test TS-sign-mag ',ts_mag, "(Sum of magnitude of sign changes)")
        print("================================"*3)
        
    ########### Generate a report on individual level ########
    
    # Only generate reports during a training phase
    if mode =="first-off":    
        # Identify the individual
        individual_numid = str(data_objects["individual"])
        
        # Load config-file
        configfile = pd.read_csv(configfilename)
        
        RUN = configfile["RUN"][0]
        F_dataset = configfile["F_dataset"][0]
        F_modelselection = configfile["F_modelselection"][0]
        F_mutation_prob = configfile["F_mutation_prob"][0]
    
        # Save information about experiment
        Inference_test["RUN"] = [RUN]*len(Inference_test)
        Inference_test["F_dataset"] = [F_dataset]*len(Inference_test)
        Inference_test["F_modelselection"] = [F_modelselection]*len(Inference_test)
        Inference_test["F_mutation_prob"] = [F_mutation_prob]*len(Inference_test)

        
        # Dump the inference tables for debugging ## REMOVED 26/09
        #Inference_test.to_csv("experiments/"+str(RUN)+"/inference_tables/Inf_test_"+individual_numid+".csv",index=False)
        
        #### Differentiate the reports: 
        if final_model == False:
            
            # Generate report with parameters of interest
            results = {"individual_numid":individual_numid,
                                   "Time":timestamp,
                                   "Traintime":time_taken,
                                   "MAE":mae_test,
                                   "MEP":mep_test,
                                   "MAEPE":maepe_test,
                                   "MAE_norm":mae_norm_test,
                                   "MEP_norm":mep_norm_test}
        
            report = dict(results, **model_params)
            
            report = pd.DataFrame(report, index=[0])
            # Storing individual
            report.to_csv("experiments/"+str(RUN)+"/individuals/"+str(RUN)+"_"+individual_numid+".csv",index=False)
            
        output = (mae_test, maepe_test, mep_test)
        
    return output



def load_evaluate(data_objects):   
    """
    Used for loading cheaper evaluation posterior to multi-objective GA-search
    """  
    # Load the settings of the current experiment:
    configfile = pd.read_csv(configfilename)
    RUN = configfile["RUN"][0]
    
    # Get the filename of the model - here decoded again, since data objects
    # will be loaded again from scratch in post-evaluation
    filename = "experiments/" + str(RUN)+ "/models/" + str(BitArray(data_objects["individual"]).uint) +".h5"
     
    #Load the model and add it to the data object
    from keras.models import load_model
    data_objects["model"] = load_model(filename)
    
    #Step3: Evaluate model using specific data object
    fitness = evaluate_model(data_objects, mode="posterior", TS = False)
    fitness = (fitness[0],fitness[1])
    ##############################################
    
    return fitness#,


def train_evaluate(individual, train_final_model=False):   
        
    # Load the settings of the current experiment:
    configfile = pd.read_csv(configfilename)
        
    RUN = configfile["RUN"][0]
    F_dataset = configfile["F_dataset"][0]
    F_modelselection = configfile["F_modelselection"][0]

    model_params = GeneConverter(individual)
            
    if train_final_model == False:
        
        #Do not include TS if this is not final model:
        eval_TS = False
        
        print("Hyper params:")
        print("================================"*3)
        print(model_params)       
        print("================================"*3)
        
        ##############################################
        
        #Step1: Dataprep
        data_objects = prepare_dataset(suffix=F_dataset, sample=1.0)
        
        #Step2: Model training
        data_objects = train_model(data_objects, model_params, final_model = train_final_model)
        
        #Step3: Evaluate model
        fitness = evaluate_model(data_objects, final_model = train_final_model, TS = eval_TS)
        
        ##############################################
        
        #Return different results based on the model selection approach
        if F_modelselection == "Single-MAE":
            fitness = (fitness[0],)
        
        if F_modelselection == "Single-MAEPE":
            fitness = (fitness[1],)
            
        if F_modelselection == "Single-MEP":
            fitness = (fitness[2],)
        
        if F_modelselection == "Multiple":
            fitness = (fitness[0],fitness[2])
            
        if F_modelselection == "RS":
            fitness = (fitness[0],fitness[1],fitness[2])
            
        print(F_modelselection+" fitness:",fitness)
 
    if train_final_model == True:
        
        #Include TS if this is final model:
        eval_TS = True
        
        print("Hyper params:")
        print("================================"*3)
        print(model_params)       
        print("================================"*3)
        
        ##############################################
        
        #Step1: Dataprep
        data_objects = prepare_dataset(suffix=F_dataset, sample=1.0)
        
        #Step2: Model training
        data_objects = train_model(data_objects, model_params, final_model=True)
        
        #Step3: Evaluate model
        fitness = evaluate_model(data_objects, final_model=True, mode="final", TS=eval_TS)
        
    return fitness#,






