# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 18:03:46 2020

@author: Mike
"""
# Set the workdir
import os
os.chdir("../")

from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from bitstring import BitArray
import time

import numpy as np
import pandas as pd

from PPM_AUTO_EVAL.Model_search_helpers import *
from PPM_AUTO_EVAL.Eval_helpers import *
from PPM_AUTO_EVAL.HPO_searchspace import *



# Load the project dir
configfile = pd.read_csv("configfile.csv")
Project_dir = configfile.Project_dir.values[0]


K = 2 ###############################################################################
DOE = ["Full_factorial","Fractional_factorial"][0]

# Load up the experiments
experiments = pd.read_csv("experiments.csv")
experiments.index = experiments.RUN.values

# Container for final results
Final_training_results = []

Max_models = np.max(experiments.RUN)


#Loop through all experiments
for experiment_i in experiments.RUN:
    
    # Bypass the experiment if it is already performed
    if experiments.Finalmodel_Done[experiment_i] == 0:
        print("="*80)
        print("Training full model for experiment: ",experiment_i)
        print("="*80)
        
        ###############################################################################
        # Store the experiment-settings for reference in the evaluation functions
        ###############################################################################
        
        #### Experiment settings
        
        # Get the number of the experiment
        RUN = experiments.RUN[experiment_i]
        
        # Load the levels of the experiment
        Experiment_Settings = np.load('Experiment_Settings.npy',allow_pickle='TRUE').item()
        
        configfile = pd.read_csv(Project_dir+"/"+str(RUN)+"/configfile.csv")
        
        F_modelselection = configfile["F_modelselection"][0]
        
        configfile.to_csv(configfilename,index=False)
        
        print("================================"*3)
        print(configfile.loc[0])
        print("================================"*3)
        
        for K_i in list(range(0,K)):
            ###############################################################################
            # Load the best K models from the experiment
            ###############################################################################
            
            if F_modelselection == "RS":
                
                Results = pd.read_csv(Project_dir+"/"+str(RUN)+"/HOF_results.csv") #"+str(RUN)+"_
                Winner = Results.sort_values(by=['MAE'], ascending=True).reset_index(drop=True).loc[K_i]
                            
                # MAE example:
                individual = np.fromstring(np.binary_repr(Winner["individual"]), dtype='S1').astype(int)
                Experiment_results = train_evaluate(individual, train_final_model=True)
            
            ###############################################################################
            if F_modelselection == "Multiple":
                
                Results = pd.read_csv(Project_dir+"/"+str(RUN)+"/HOF_results.csv") #"+str(RUN)+"_
                
                # Selection according to MAE or MAEPE:
                Pareto_winner = Results.loc[K_i]
                
                # Pareto:
                individual = np.fromstring(np.binary_repr(Pareto_winner["individual"]), dtype='S1').astype(int)
                Experiment_results = train_evaluate(individual, train_final_model=True)
                
            
            ###############################################################################
            if F_modelselection == "Single-MAE" or F_modelselection == "Single-MEP":
                
                Results = pd.read_csv(Project_dir+"/"+str(RUN)+"/HOF_results.csv") #"+str(RUN)+"_
                
                if F_modelselection == "Single-MAE":
                    # Selection according to MAE or MAEPE:
                    Winner = Results.sort_values(by=['MAE'], ascending=True).reset_index(drop=True).loc[K_i]
                    individual = np.fromstring(np.binary_repr(Winner["individual"]), dtype='S1').astype(int)
                    Experiment_results = train_evaluate(individual, train_final_model=True)
                    
                if F_modelselection == "Single-MEP":
                    # Selection according to MAE or MAEPE:
                    Winner = Results.sort_values(by=['MEP'], ascending=True).reset_index(drop=True).loc[K_i]
                    individual = np.fromstring(np.binary_repr(Winner["individual"]), dtype='S1').astype(int)
                    Experiment_results = train_evaluate(individual, train_final_model=True)
            
            ###############################################################################
            #Store the training results for each model
            
            Experiment_results["RUN"] = experiment_i
            Experiment_results.to_csv(Project_dir+"/Final_models/individuals/"+"Final_" + str(RUN) + "_"+ str(K_i) + ".csv",index=False)
                
            """
            if len(Final_training_results) > 0:
                Final_training_results = Final_training_results.append(Experiment_results)            
                Final_training_results.to_csv(Project_dir+"/"+"Final_models/Final_model_training_results.csv",index=False)
                            
            if len(Final_training_results) == 0:
                Final_training_results = Experiment_results
                Final_training_results.to_csv(Project_dir+"/"+"Final_models/Final_model_training_results.csv",index=False)
            """
            
            ###############################################################################
            # Log the status of the experiment
            print("================================"*3)
            print("Finished experiment",RUN)
            print("================================"*3)
            experiments.Finalmodel_Done.loc[experiments.RUN == experiment_i] = 1
            experiments.to_csv("experiments.csv",index=False)
            
###############################################################################
# Collect all final models

if RUN == Max_models:
    """
    Safe way to combine all models with experiments:
    """
    
    # Get all files in the final models folder
    path = Project_dir+"/Final_models/individuals/"
    from os import listdir
    from os.path import isfile, join
    
    final_models = [f for f in listdir(path) if isfile(join(path, f))]
    Training_results = []
    
    for file in final_models:
        model_i = pd.read_csv(path+file)
        
        #Get the RUN id for later merge
        import re
      
        
        RUN = re.search('Final_(.*).csv', file)[1]
        
        """
        
        if len(RUN) > 2:
            full_id = re.search('Final_(.*).csv', file)[1]
            RUN = re.search('(.*)_', full_id)[1]
        """  
        print(RUN)
        
        model_i["RUN"] = int(RUN)
        
        if len(RUN) < 3:
            Training_results.append(model_i)
    
    # Combine to DF
    Training_results_df = pd.concat(Training_results)
    Training_results_df["RUN"] = Training_results_df["RUN"].astype(int)
    
    # Merge to experiment table
    #experiments_merged = Training_results_df.merge(experiments, on='RUN', how='outer')
    
    # Merge to experiment table
    experiments_merged = experiments.merge(Training_results_df, on='RUN', how='left')
    
    # Export to final table
    experiments_merged.to_csv(Project_dir+"/"+"Final_experiments_merged.csv",index=False)

