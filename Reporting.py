# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 19:22:15 2020

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

from Eval_helpers import *


"""
###############################################################################
# REPORTING FUNCTIONS
###############################################################################
"""


def SaveLastGenResults(data_objects, res, logbook, RUN, start_time):

    # Wrap chaper evaluation into a function
    def post_evaluation(data_objects, individual):
        # Add specific individual to data object
        data_objects["individual"] = individual
        
        # Evaluate model
        fitness = load_evaluate(data_objects)
        return fitness
    
    #Inspect the logbook:
    print(logbook)
    
    # Print top N solutions - (1st only, for now)
    best_individuals = tools.selBest(res,k = 1)
    
    #print("fitness",post_evaluation(data_objects, best_individuals[0]))
    
    
    # Fetch the individuals in the last generation
    last_gen = res
    
    for i in range(0,len(last_gen)):
        # Convert to numbers
        individual_numid = str(BitArray(last_gen[i]).uint)
        
        if i == 0:
            # Load each result table
            last_gen_results = pd.read_csv("experiments/"+str(RUN)+"/individuals/"+str(RUN)+"_"+individual_numid+".csv") #+RUN+"_"
            
        if i > 0:
            # Load each result table
            individual_i_res = pd.read_csv("experiments/"+str(RUN)+"/individuals/"+str(RUN)+"_"+individual_numid+".csv")
            
            #append to existing table
            last_gen_results = last_gen_results.append(individual_i_res, ignore_index=True)
    
    # Store the amount of time the experiment took
    end_time = time.time()
    Time_sec = end_time - start_time
    
    last_gen_results["Total_time"] = [Time_sec]*len(last_gen_results)
    
    # Append it to the full generation results
    last_gen_results["Search"] = "GA"
    last_gen_results.to_csv("experiments/"+str(RUN)+"/HOF_results.csv",index=False)
    print("Saved HOF results")
    
    return last_gen_results





def SaveLogbookResults(logbook,F_modelselection, RUN, population_size, num_generations, gene_length):

    #for gen in 
    len(logbook)
    
    generation=[]
    evals=[]
    acc_avg=[]
    ear_avg=[]
    acc_std=[]
    ear_std=[]
    acc_min=[]
    ear_min=[]
    acc_max=[]
    ear_max=[]
    
    if F_modelselection == "Multiple":
        for gen in range(0,len(logbook)):
            generation.append(gen+1)
            evals.append(logbook[gen]["nevals"])
            acc_avg.append(logbook[gen]["avg"][0])
            ear_avg.append(logbook[gen]["avg"][1])
            acc_std.append(logbook[gen]["std"][0])
            ear_std.append(logbook[gen]["std"][1])
            
            acc_min.append(logbook[gen]["min"][0])
            ear_min.append(logbook[gen]["min"][1])
            acc_max.append(logbook[gen]["max"][0])
            ear_max.append(logbook[gen]["max"][1])
        
        logbook_results = pd.DataFrame({"gen":generation,"evals":evals,
                                        "acc_avg":acc_avg,
                                        "acc_std":acc_std,
                                        "acc_min":acc_min,
                                        "acc_max":acc_max,
                                        "ear_avg":ear_avg,
                                        "ear_std":ear_std,
                                        "ear_min":ear_min,
                                        "ear_max":ear_max})
        
        logbook_results.to_csv("experiments/"+str(RUN)+"/"+str(RUN)+"_GA_logbook_results.csv",index=False)
        
    if F_modelselection == "Single-MAE":
        for gen in range(0,len(logbook)):
            generation.append(gen+1)
            evals.append(logbook[gen]["nevals"])
            acc_avg.append(logbook[gen]["avg"])
            acc_std.append(logbook[gen]["std"])
            
            acc_min.append(logbook[gen]["min"])
            acc_max.append(logbook[gen]["max"])
           
        logbook_results = pd.DataFrame({"gen":generation,
                                        "evals":evals,
                                        "acc_avg":acc_avg,
                                        "acc_std":acc_std,
                                        "acc_min":acc_min,
                                        "acc_max":acc_max})
    
        logbook_results.to_csv("experiments/"+str(RUN)+"/"+str(RUN)+"_GA_logbook_results.csv",index=False)
    
    if F_modelselection == "Single-MAEPE":
        for gen in range(0,len(logbook)):
            generation.append(gen+1)
            evals.append(logbook[gen]["nevals"])
            ear_avg.append(logbook[gen]["avg"])
            ear_std.append(logbook[gen]["std"])
            
            ear_min.append(logbook[gen]["min"])
            ear_max.append(logbook[gen]["max"])
           
        logbook_results = pd.DataFrame({"gen":generation,
                                        "evals":evals,
                                        "ear_avg":ear_avg,
                                        "ear_std":ear_std,
                                        "ear_min":ear_min,
                                        "ear_max":ear_max})
    
        logbook_results.to_csv("experiments/"+str(RUN)+"/"+str(RUN)+"_GA_logbook_results.csv",index=False)
    
    if F_modelselection == "RS":
        # Store starttime
        start_time = time.time()
        
        # Determine number of model searches
        
        n_models =  population_size * (num_generations+1)
        
        list_of_models = []
        
        # Run n random search iterations
        for i in range(0,n_models):
            print("model ",str(i))
            
            #Generate a random individual
            individual = np.random.randint(2, size=(gene_length,))
            
            #Store it on the list of models
            list_of_models.append(individual)
            
            # Train a model based on the individual:
            ACC, EAR = train_evaluate(individual)
            
        # When done, log the results
        for i in range(0,len(list_of_models)):
            # Convert to numbers
            individual_numid = str(BitArray(list_of_models[i]).uint)
            
            if i == 0:
                # Load each result table
                last_gen_results = pd.read_csv("experiments/"+str(RUN)+"/individuals/"+str(RUN)+"_"+individual_numid+".csv")
                
            if i > 0:
                # Load each result table
                individual_i_res = pd.read_csv("experiments/"+str(RUN)+"/individuals/"+str(RUN)+"_"+individual_numid+".csv")
                
                #append to existing table
                last_gen_results = last_gen_results.append(individual_i_res, ignore_index=True)
        
        # Store the amount of time the experiment took
        end_time = time.time()
        Time_sec = end_time - start_time
        
        last_gen_results["Total_time"] = [Time_sec]*len(last_gen_results)
        
        # Append it to the full generation results
        last_gen_results.to_csv("experiments/"+str(RUN)+"/"+str(RUN)+"_RS_all_models_results.csv",index=False)
    
    return print("Saved last gen results")

def StoreParetoFronts(data_objects, res, RUN):
    from Model_search_helpers import load_evaluate
    
    # Wrap chaper evaluation into a function
    def post_evaluation(data_objects, individual):
        # Add specific individual to data object
        data_objects["individual"] = individual
        
        # Evaluate model
        fitness = load_evaluate(data_objects)
        return fitness
    
    #Get the non-dominated solutions
    fronts = tools.emo.sortLogNondominated(res, len(res))
    len(fronts)
    
    
    # Get max top 3 pareto fronts
    #if len(fronts) > 3:
    #    length=3
    #if len(fronts) < 4:
    length=len(fronts)
    
    # Store pareto fronts in table
    front_no = []
    solution_no = []
    individual = []
    accuracy=[]
    earliness=[]
    
    #Loop over the fronts
    for frontno in range(0,length):
        for i in range(0,len(fronts[frontno])):
            #get fitness of each solution on first front
            
            fitness = post_evaluation(data_objects,fronts[frontno][i])
            individual.append(BitArray(fronts[frontno][i]).uint)
            
            front_no.append("Front "+str(frontno+1))
            solution_no.append(i)
            accuracy.append(fitness[0])
            earliness.append(fitness[1])
    
    pareto_fronts = pd.DataFrame({"front_no":front_no,"solution_no":solution_no,
                                  "individual":individual,"accuracy":accuracy,
                                  "earliness":earliness})
    
    #Drop duplicate individuals:
    pareto_fronts = pareto_fronts.drop_duplicates(subset='individual', keep="first")
    
    #Store the pareto fronts
    pareto_fronts.to_csv("experiments/"+str(RUN)+"/"+str(RUN)+"_GA_pareto_fronts.csv",index=False)

    
    ###############################################################################
    # Save plot of the results of the pareto front
    ###############################################################################
    #https://www.youtube.com/watch?v=Hm2LK4vJzRw #25:00
    
    import matplotlib.pyplot as plt
    
    
    groups = pareto_fronts.groupby('front_no')
    
    # Plot
    fig, ax = plt.subplots(1, figsize=(10,10))
    
    
    #ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    
    for name, group in groups:
        ax.plot(group.accuracy.values, 
                group.earliness.values, 
                marker='o', linestyle='', ms=10, label=name)
        
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height* 0.8])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.xlabel('Objective 1: MAE')
    plt.ylabel('Objective 2: MEP')
    
    #plt.show()
    plt.savefig("experiments/"+str(RUN)+"/"+str(RUN)+"_GA_pareto_fronts.png",dpi=150)
    
    return print("Saved pareto fronts (log and plot)..")