# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 16:05:06 2020
@author: Mike
"""

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from Model_search_helpers import *

from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from bitstring import BitArray
import time

##############################################################################
# Specify number of nodes to work at the same time.

v_cores = 8

import multiprocessing
from joblib import Parallel, delayed
    

pool = multiprocessing.Pool(processes=v_cores) #multiprocessing.cpu_count())

##############################################################################

"""
# For experimentation only:
    

case = np.asarray([1,1,1,1,1,1,1,1,1,1,2,2,2,2,2])
t = np.asarray([1,2,3,4,5,5,6,7,8,10, 1,2,3,4,5])

T = np.asarray([10,10,10,10,10,10,10,10,10,10, 5,5,5,5,5])



y = np.asarray([10, 8, 8, 7, 5, 5, 3, 1, 1, 1,  5, 5, 3, 1, 1])

y_hat = np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1]) 

resid = y - y_hat

N = 2

factor = (T-t)/t#((T-t)/t)

# Metrics:

AE = np.abs(y - y_hat)
AE_norm = (AE - np.min(AE)) / (np.max(AE) - np.min(AE))

EP = np.abs(y - y_hat)**((T-t)/T)
EP_norm = (EP - np.min(EP)) / (np.max(EP) - np.min(EP))


# Table:
    
Table = pd.DataFrame({"Case":case,"t":t, "T":t,"N":N, "y":y, "y_hat":y_hat,"Residual":resid, 
                      "AE":AE, "EP":EP, "AE_norm":AE_norm, "EP_norm":EP_norm})


MEP = np.mean(EP)

MAE = np.mean(AE)

lambda_weight = 1

MAEPE = np.mean(AE_norm) + lambda_weight*np.mean(EP_norm)
MAEPE

lambdas = [0,0.5, 1, 1.5,2]

for i in lambdas:
    print(np.mean(AE_norm) + i*np.mean(EP_norm))

"""
"""
Todo: Make functionality that generates all the needed variables, if they are not present in inference table.

Then make function for each of the metrics to be computed.

!! important: What to do when last event is left out? 
    Remove_last_evt == True
        t = t-1
        T = T-1
        
    Remove_last_evt == False
        t = t    
"""
"""

## load inference tables for evaluation and analysis
suffix = "Drop_last_ev/SF"
Inference_train = pd.read_csv("data/"+suffix+"_Inference_train.csv")
Inference_test = pd.read_csv("data/"+suffix+"_Inference_test.csv")


# Inference table metrics:

T = "num_events"
t = "event_number"

case = "caseid"
y = "y"
y_pred = "y_pred"
"""
# Event-level metrics:

def AE(y, y_hat):
    AE = np.abs(y - y_hat)
    return AE

def EP(y, y_hat, t, T):
    EP = np.abs(y - y_hat)**((T-t)/T)
    return EP

def NORM(Vector):
    norm = (Vector - np.min(Vector)) / (np.max(Vector) - np.min(Vector))
    return norm

# Aggregated metrics:
    
# MAE = np.mean(AE)
# MEP = np.mean(EP)
    
def MAEPE(AE_norm, EP_norm, lambda_weight):
    MAEPE = np.mean(AE_norm) + lambda_weight*np.mean(EP_norm)
    return MAEPE

"""
# Test the metrics:
    
Inference_test["y_pred"] = Inference_test["y"] * 1.2
    
Inference_test["AE"] = AE(Inference_test["y"], Inference_test["y_pred"])

Inference_test["EP"] = EP(Inference_test["y"], Inference_test["y_pred"], t=Inference_test["event_number"], T=Inference_test["num_events"])


# MAE:
MAE = np.mean(Inference_test["AE"])
Inference_test["AE_norm"] = NORM(Inference_test["AE"])

# MEP
MEP = np.mean(Inference_test["EP"])
Inference_test["EP_norm"] = NORM(Inference_test["EP"])

#MAEPE
weight = 1
MAEPE_out = MAEPE(AE_norm=Inference_test["AE_norm"], EP_norm=Inference_test["EP_norm"] , lambda_weight=weight)


lambdas = [0,0.5, 1, 1.5,2]

for i in lambdas:
    print(MAEPE(AE_norm=Inference_test["AE_norm"], EP_norm=Inference_test["EP_norm"] , lambda_weight=i))


"""



##############################################################################


def Sign1st(y0,y1):
    firstdiff = y1 - y0
    sign = np.sign(firstdiff)
    return sign

def Earliness(table):
    """
    Evaluates earliness alone
    
    Input: 
        Inference table with relevant variables
        
            T = "num_events"
            t = "event_number"
            
            case = "caseid"
            y = "y"
            y_pred = "y_pred"
    
    Output:
        EP, AE, MEP, MAE, MAEPE
    """
    
    # Calculate metrics:
        
    #Absolute error
    table["AE"] = AE(table["y"], table["y_pred"])

    #Eearliness penalty
    table["EP"] = EP(table["y"], table["y_pred"], t=table["event_number"], T=table["num_events"])    
    
    # MAE:
    MAE = np.mean(table["AE"])
    table["AE_norm"] = NORM(table["AE"])
    MAE_norm = np.mean(table["AE_norm"])
    
    # MEP
    MEP = np.mean(table["EP"])
    table["EP_norm"] = NORM(table["EP"])
    MEP_norm = np.mean(table["EP_norm"])
    
    #MAEPE
    weight = 1
    MAEPE_out = MAEPE(AE_norm=table["AE_norm"], EP_norm=table["EP_norm"] , lambda_weight=weight) 
   
    #Aggregated metrics:
    table["MAE"] = MAE
    table["MEP"] = MEP
    table["MEP_norm"] = MEP_norm
    table["MAE_norm"] = MAE_norm
    table["MAEPE"] = MAEPE_out
    
    return table



def TS_group(table):
    #print("Evaluating earliness..")
    #Look at only the case
    case = table
    
    #reset the index, so that it can be used as the time t variable
    case.index = range(1,len(case)+1)
    
    case_id = case.caseid[1]
    
    """
    Temporal stability metrics
    """
    table = table.reset_index(drop=True)
    #Make placeholders
    table["1stdiff"] = 0
    table["sign"] = 0
    table["sign_diff"] = 0
    table["sign_magnitude"] = 0
       
    #Loop through each prefix
    for i in range(0,len(table)):
        
        #If its the first prefix, set everything to 0
        if i == 0:
            sign_i = Sign1st(table.loc[i,"y_pred"], 0)
            table.loc[i,"sign"] = sign_i
            table["sign_diff"] = 0

            table.loc[i,"sign_magnitude"] = 0
            table.loc[i,"1stdiff"] = 0

        #If its not the first, do the calculations
        if i > 0:
            #First difference
            table.loc[i,"1stdiff"] = table.loc[i-1,"y_pred"] - table.loc[i,"y_pred"]

            #Sign of the first difference
            sign_i = Sign1st(table.loc[i-1,"y_pred"], table.loc[i,"y_pred"])
            table.loc[i,"sign"] = sign_i

            #Dummy to indicate if there is a difference between the signs
            table.loc[i,"sign_diff"] = (table.loc[i,"sign"] != table.loc[i-1,"sign"])*1

            #Calculate the absolute difference, when there is a sign change
            table.loc[i,"sign_magnitude"] = np.abs(table.loc[i,"sign_diff"] * (table.loc[i,"y_pred"] - table.loc[i-1,"y_pred"])) 

    # Sum of sign changes
    SSC = np.sum(table["sign_diff"])
    # Proportion of sign changes
    PSC = np.sum(table["sign_diff"])/len(table)
    # Sum of magnitude of sign changes
    SMSC = np.sum(table["sign_magnitude"])
    
    return SSC, PSC, SMSC, case_id #MAEPE, MAE, 



def applyParallel(dfGrouped, func):
    JOBZ = 12 #multiprocessing.cpu_count()
    retLst = Parallel(n_jobs=JOBZ)(delayed(func)(group) for name, group in dfGrouped)
    return retLst #pd.concat(retLst)



def Earliness_and_TS(Inference_test, parallel=False, EAR=True, TS=True):
    
    start_time = time.time()
    
    if TS == True:
        """
        Temporal stability
        """
        print("Temporal stability...")
        if parallel == True:
            res = applyParallel(Inference_test.groupby(["caseid"]), TS_group)
                
        if parallel == False:
            values = Inference_test.groupby(["caseid"]).apply(TS_group)
        
        cols = ["SSC", "PSC", "SMSC", "caseid"] #"MAEPE","MAE",
        res=pd.DataFrame(values.tolist())#, index=values.index
        res.columns = cols
        
        Inference_test = Inference_test.merge(res, on='caseid', how='left')
        
    if EAR == True:        
        """
        Earliness
        """
        print("Earliness...")
        Inference_test = Earliness(Inference_test)
        
        end_time = time.time()
        Time_sec = end_time - start_time
        print(Time_sec)

    return Inference_test


"""
start_time = time.time()
#groupby approach:
values = Inference_test.groupby(["caseid"]).apply(Earliness_group)#(lambda group: Earliness_group(group))

end_time = time.time()
Time_sec = end_time - start_time
print(Time_sec)

cols = ["MAEPE","MAE","SSC", "PSC", "SMSC", "caseid"]
res=pd.DataFrame(values.tolist())#, index=values.index
res.columns = cols

INF_test = Inference_test.merge(res, on='caseid', how='left')
"""


def TemporalDiff(df):
    """
    Computes temporal difference variables for remaining time model evaluation
    
    Parameters
    ----------
    df : inference table with standard colummns
            - caseid (identifier for a trace)
            - y_pred (the predicted remaining time at each event in the trace)
    Returns
    -------
    out : inference table with error measures
            - 1stdiff (first difference between two predictions)
            - sign (the direction of the first difference)
            - sign_diff (if there is a difference between signs at time t and t-1)
            - sign_magnitude (the sum of direction change between time t and t-1)
    """
    caseids = df['caseid'].unique()
    out = []
    print("Evaluating Temporal stability..")
    
    for idx in caseids:
        # Subset every caseid
        table = df.loc[df["caseid"]==idx]
        table = table.reset_index(drop=True)
        
        #Make placeholders
        table["1stdiff"] = 0
        table["sign"] = 0
        table["sign_diff"] = 9
        table["sign_magnitude"] = 9
        
        #Loop through each prefix
        for i in range(0,len(table)):
            
            #If its the first prefix, set everything to 0
            if i == 0:
                sign_i = Sign1st(table.loc[i,"y_pred"], 0)
                table.loc[i,"sign"] = sign_i
                table["sign_diff"] = 0

                table.loc[i,"sign_magnitude"] = 0
                table.loc[i,"1stdiff"] = 0

            #If its not the first, do the calculations
            if i > 0:
                #First difference
                table.loc[i,"1stdiff"] = table.loc[i-1,"y_pred"] - table.loc[i,"y_pred"]

                #Sign of the first difference
                sign_i = Sign1st(table.loc[i-1,"y_pred"], table.loc[i,"y_pred"])
                table.loc[i,"sign"] = sign_i

                #Dummy to indicate if there is a difference between the signs
                table.loc[i,"sign_diff"] = (table.loc[i,"sign"] != table.loc[i-1,"sign"])*1

                #Calculate the absolute difference, when there is a sign change
                table.loc[i,"sign_magnitude"] = np.abs(table.loc[i,"sign_diff"] * (table.loc[i,"y_pred"] - table.loc[i-1,"y_pred"])) 
            
        if len(out) != 0:
            out = pd.concat([out,table],axis=0)
        
        if len(out) == 0:
            out = table
            
    return out

def find_UCL(table):
    """
    This function finds the UCL from an inference table
    """
    print("Finding UCL.")
    
    durations = []
    ids = []
    
    sigmas = 3
    
    for idx in set(table.caseid):
        #print(idx)
        
        #Look at only the case
        case = table.loc[table["caseid"] == idx]
        
        #Get remaining time from ground truth
        duration_max = np.max(case["y"].values)
        
        #append to list of ids and all durations
        durations.append(duration_max)
        ids.append(idx)
    
    #Store results in table
    df = pd.DataFrame({"caseid":ids,"duration":durations})

    #Get six sigma statistics:
    mean_dur = np.mean(df["duration"].values)
    std_dur = np.std(df["duration"])
    
    #Six sigma
    UCL = mean_dur+(std_dur*sigmas)
    LCL = mean_dur-(std_dur*sigmas)
    
    return UCL




def UCL_eval(table, UCL):
    """
    This function takes the duration from the test period, and:
        1) Flags cases that exceeded UCL
        2) Flags cases with predicted UCL violation
    
    """
    
    print("Evaluating UCL conditions..")
    durations = []
    ids = []
        
    for idx in set(table.caseid):
        #print(idx)
        
        #Look at only the case
        case = table.loc[table["caseid"] == idx]
        
        #Get remaining time from ground truth
        duration_max = np.max(case["y"].values)
        
        #append to list of ids and all durations
        durations.append(duration_max)
        ids.append(idx)
    
    #Store results in table
    df = pd.DataFrame({"caseid":ids,
                       "duration":durations}) #case duration

    
    ####### evaluation table ##########
    df["UCL_violation"] = (df["duration"] > UCL)*1
    
    print("UCL violations:",str(np.sum(df["UCL_violation"])))
    
    
    # Loop through the cases
    for idx in set(table.caseid):
        
        #Subset the case only
        case = table.loc[table["caseid"] == idx]
                
        
        """ 
        ############### ############## ##########
        #Get accumulated elapsed time + RT
        
        This will enable step-wise evaluation 
        
        This is the next step in UCL eval functionality
        ############### ############## ##########
        """
        #############################################
        
        # Go through each event
        
        #UCL = 81
        
        times = case.y_t.values
        predictions = case.y_pred.values
        
        time_taken = np.cumsum(times)
        
        est_duration = time_taken + predictions
        
        pred_violations = (est_duration >= UCL)*1
        
        # Get the ground truth for the UCL
        duration = np.sum(times)
        
        y = (duration >= UCL)*1
        
        actual_violation = [y]*len(case)
        
        # Predicted
        table.loc[table.caseid == idx,"y_pred_UCL_viol_ev"] = pred_violations
        # Actual
        table.loc[table.caseid == idx,"y_UCL_viol_ev"] = actual_violation
        
        #############################################
        
        # Do it on case-level
        
        #Get the max predicted duration, per case
        max_predicted_RT = np.max(case["y_pred"])
        
        #Is it greater than or equal to UCL?
        Pred_UCL_VIOL = (max_predicted_RT >= UCL)*1
        
        
        #Was it really a UCL violation? Look in df made earlier
        UCL_GT = df.loc[df["caseid"] == idx]
        UCL_VIOL = UCL_GT["UCL_violation"].values
    
        # Log the UCL violations in the final output table
        
        # Predicted
        table.loc[table.caseid == idx,"y_pred_UCL_viol"] = [Pred_UCL_VIOL]*len(case)
        # Actual
        table.loc[table.caseid == idx,"y_UCL_viol"] = [UCL_VIOL]*len(case)
        
    #Fix formats before return
    table['y_pred_UCL_viol'] = table['y_pred_UCL_viol'].astype(int)
    table['y_UCL_viol'] = table['y_UCL_viol'].astype(int)
    
    table['y_pred_UCL_viol_ev'] = table['y_pred_UCL_viol'].astype(int)
    table['y_UCL_viol_ev'] = table['y_UCL_viol'].astype(int)
    
    return table



def UCL_eval_class(table, level="case"):
    """
    Takes the UCL evaluations and outputs classification stats
    """
    if level == "case":
        y_pred = table["y_pred_UCL_viol"].values
        y_true = table["y_UCL_viol"].values

    if level == "event":
        y_pred = table["y_pred_UCL_viol_ev"].values
        y_true = table["y_UCL_viol_ev"].values

    
    print("=======================================")
    print("Actual UCL violations:",str(np.sum(y_true)))
    print("Predicted UCL violations:",str(np.sum(y_pred)))
    
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_true, y_pred)
    print('Accuracy: %f' % accuracy)
    
    # precision tp / (tp + fp)
    precision = precision_score(y_true, y_pred)
    print('Precision: %f' % precision)
    
    # recall: tp / (tp + fn)
    recall = recall_score(y_true, y_pred)
    print('Recall: %f' % recall)
    
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_true, y_pred)
    print('F1 score: %f' % f1)
    print("=======================================")
    
    return (precision, recall, accuracy, f1, table)




"""
data_objects = prepare_dataset(suffix="SF_eventlog_filter_length1", sample=1.0)
Inference_train = data_objects["Inference_train"]
Inference_test = data_objects["Inference_test"]
Inference_test["y_pred"] = Inference_test["y_t"].values*5
# Find UCL in train table (or window)
UCL = find_UCL(Inference_train)
# Mark violations
Inference_test = UCL_eval(Inference_test, UCL)
# Get fitness stats
UCL_eval_class(Inference_test)
"""

