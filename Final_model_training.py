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


K = 1
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
        
        """
        if DOE == "Full_factorial":
            
            # Convert the factors into the original level values
            F_modelselection = Experiment_Settings["F_modelselection"][int(experiments.F_modelselection[RUN])]
            F_dataset = Experiment_Settings["F_dataset"][int(experiments.F_dataset[RUN])]
            F_mutation_prob = experiments.F_mutation_prob[RUN]
            F_lofi_epochs = experiments.F_lofi_epochs[RUN]
            F_modeltype = "LSTM" #Experiment_Settings["F_modeltype"][int(experiments.F_modeltype[RUN])]
            
            F_num_generations = int(experiments.F_num_generations[RUN])
            F_population_size = int(experiments.F_population_size[RUN])
            F_lofi_epochs = int(experiments.F_lofi_epochs[RUN])
 
        if DOE == "Fractional_factorial":
            
            #Load without conversion
            F_modelselection = experiments.F_modelselection[RUN]
            F_dataset = experiments.F_dataset[RUN]
            F_mutation_prob = experiments.F_mutation_prob[RUN]
            F_lofi_epochs = experiments.F_lofi_epochs[RUN]
            F_modeltype = "LSTM" #experiments.F_modeltype[RUN]
            
            F_num_generations = experiments.F_num_generations[RUN]
            F_population_size = experiments.F_population_size[RUN]
            F_lofi_epochs = experiments.F_lofi_epochs[RUN]
            
           
               
        # Search parameters:        
        population_size = F_population_size
        num_generations = F_num_generations
        k_in_hall_of_fame = population_size * num_generations #Store all individuals
        Finalmodel_epochs = 700

        
        #### Experiment settings
                
        configfile = pd.DataFrame({"RUN":RUN,
                                   "F_modelselection":F_modelselection,
                                   "F_dataset":F_dataset,
                                   "F_mutation_prob":F_mutation_prob,
                                   "F_num_generations":F_num_generations,
                                   "F_population_size":F_population_size,
                                   "F_lofi_epochs":F_lofi_epochs,
                                   "F_modeltype":F_modeltype,
                                   "HOF_size":k_in_hall_of_fame,
                                   "Finalmodel_epochs":Finalmodel_epochs}, index=[0])
        
        configfile.to_csv(configfilename,index=False)
        configfile.to_csv("experiments/"+str(RUN)+"/"+"Configfile.csv",index=False)
        """ 
        print("================================"*3)
        print(configfile.loc[0])
        print("================================"*3)
        
        
        ###############################################################################
        # Load the best K models from the experiment
        ###############################################################################
        
        if F_modelselection == "RS":
            
            Results = pd.read_csv(Project_dir+"/"+str(RUN)+"/HOF_results.csv") #"+str(RUN)+"_
            Winner = Results.sort_values(by=['MAE'], ascending=True).reset_index(drop=True).loc[0]
                        
            # MAE example:
            individual = np.fromstring(np.binary_repr(Winner["individual"]), dtype='S1').astype(int)
            Experiment_results = train_evaluate(individual, train_final_model=True)
        
        ###############################################################################
        if F_modelselection == "Multiple":
            
            Results = pd.read_csv(Project_dir+"/"+str(RUN)+"/HOF_results.csv") #"+str(RUN)+"_
            
            # Selection according to MAE or MAEPE:
            Pareto_winner = Results.loc[0]
            
            # Pareto:
            individual = np.fromstring(np.binary_repr(Pareto_winner["individual"]), dtype='S1').astype(int)
            Experiment_results = train_evaluate(individual, train_final_model=True)
            
        
        ###############################################################################
        if F_modelselection == "Single-MAE" or F_modelselection == "Single-MEP":
            
            Results = pd.read_csv(Project_dir+"/"+str(RUN)+"/HOF_results.csv") #"+str(RUN)+"_
            
            if F_modelselection == "Single-MAE":
                # Selection according to MAE or MAEPE:
                Winner = Results.sort_values(by=['MAE'], ascending=True).reset_index(drop=True).loc[0]
                individual = np.fromstring(np.binary_repr(Winner["individual"]), dtype='S1').astype(int)
                Experiment_results = train_evaluate(individual, train_final_model=True)
                
            if F_modelselection == "Single-MEP":
                # Selection according to MAE or MAEPE:
                Winner = Results.sort_values(by=['MEP'], ascending=True).reset_index(drop=True).loc[0]
                individual = np.fromstring(np.binary_repr(Winner["individual"]), dtype='S1').astype(int)
                Experiment_results = train_evaluate(individual, train_final_model=True)
        
        ###############################################################################
        #Store the training results for each model
        
        Experiment_results["RUN"] = experiment_i
            
        if len(Final_training_results) > 0:
            Final_training_results = Final_training_results.append(Experiment_results)            
            Final_training_results.to_csv(Project_dir+"/"+"Final_models/Final_model_training_results.csv",index=False)
                        
        if len(Final_training_results) == 0:
            Final_training_results = Experiment_results
            Final_training_results.to_csv(Project_dir+"/"+"Final_models/Final_model_training_results.csv",index=False)
            
        ###############################################################################
        # Log the status of the experiment
        print("================================"*3)
        print("Finished experiment",RUN)
        print("================================"*3)
        experiments.Finalmodel_Done.loc[experiments.RUN == experiment_i] = 1
        experiments.to_csv("experiments.csv",index=False)
            
        
if RUN == Max_models:
    #Create the final table:
    Final_training_results = pd.read_csv(Project_dir+"/"+"Final_models/Final_model_training_results.csv")
    experiments_merged = experiments.merge(Final_training_results, on='RUN', how='left')
    experiments_merged.to_csv(Project_dir+"/"+"Final_experiments_merged.csv",index=False)

