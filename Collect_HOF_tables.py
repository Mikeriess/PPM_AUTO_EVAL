# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 10:46:02 2020

@author: mireiss
"""


import pandas as pd
import numpy as np


folderlist = list(range(1,8+1))
folderlist

All_HOF = []

for folder in folderlist:
    configurations = pd.read_csv(str(folder)+"/"+"Configfile.csv")
    
    #if configurations["F_modelselection"].loc[0] == 'Single-MAE':
    
    HOF = pd.read_csv(str(folder)+"/HOF_results.csv")
    
    HOF["RUN"] = folder
    HOF["F_mutation_prob"] = configurations["F_mutation_prob"].loc[0]
    HOF["F_num_generations"] = configurations["F_num_generations"].loc[0]
    HOF["F_population_size"] = configurations["F_population_size"].loc[0]
    HOF["F_lofi_epochs"] = configurations["F_lofi_epochs"].loc[0]
    HOF["HOF_size"] = configurations["HOF_size"].loc[0]

    All_HOF.append(HOF)
    

All_HOF = pd.concat(All_HOF)

All_HOF.to_csv("../ALL_HOF_files.csv",index=False)