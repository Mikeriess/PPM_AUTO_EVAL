# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 10:30:53 2020

@author: mireiss
"""

import pandas as pd
import numpy as np


folderlist = list(range(1,8+1))
folderlist

All_logs = []

for folder in folderlist:
    configurations = pd.read_csv(str(folder)+"/"+"Configfile.csv")
    
    if configurations["F_modelselection"].loc[0] == 'Single-MAE':
        
        logbook = pd.read_csv(str(folder)+"/"+str(folder)+"_GA_logbook_results.csv")
        
        logbook["RUN"] = folder
        logbook["F_mutation_prob"] = configurations["F_mutation_prob"].loc[0]
        logbook["F_num_generations"] = configurations["F_num_generations"].loc[0]
        logbook["F_population_size"] = configurations["F_population_size"].loc[0]
        logbook["F_lofi_epochs"] = configurations["F_lofi_epochs"].loc[0]
        logbook["HOF_size"] = configurations["HOF_size"].loc[0]
    
        All_logs.append(logbook)
        

All_Logbooks = pd.concat(All_logs)
len(All_logs)
All_Logbooks.to_csv("../ALL_Logbooks.csv",index=False)