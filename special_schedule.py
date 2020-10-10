# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 23:55:14 2020

@author: Mike
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:39:58 2020

@author: Mike
"""
import os 

##############################################################################
# Project specs:

# Project name
project_name = "PPM-AUTO-EVAL_helpdesk" #_helpdesk

# Parent destination    
parent = "A:/EXPERIMENTS/"

# Generate the project folder if it doesnt exist
path = os.path.join(parent, str(project_name))
if not os.path.exists(path):
    os.mkdir(path)
    os.mkdir(path+"/experiments")

##############################################################################

# Specify which configfile to store:
configfilename = "configfile.csv"
store_progress = False

# Parent Directory path for storage of experiments
parent_dir = path+"/experiments"

# Specify whether to use Fractional factorial (with labels) or full factorial without.
DOE = ["Full_factorial","Fractional_factorial"][0]

##############################################################################


from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from bitstring import BitArray
import time

import numpy as np
import pandas as pd
from Model_search_helpers import *
from Eval_helpers import *
from Reporting import*

from HPO_searchspace import *

# Load up the experiments
experiments = pd.read_csv("experiments.csv")
experiments.index = experiments.RUN.values

experiment_list = experiments.RUN.values

""" GA settings:
Elitism: https://groups.google.com/forum/#!topic/deap-users/FCPOYmO_enQ
         https://groups.google.com/forum/#!topic/deap-users/H6qSjra11GE
"""

# Load constant GA search settings
elitism = 1                  #same as GA paper
#crossover_probability = 0.8 #same as GA paper <<<<<< changed to 1-mutation prob, since they either mutate or cross over


#Loop through all experiments in the initial list
for experiment_i in experiment_list:
    print("================================"*3)
    print("Starting experiment: ",experiment_i)
        
    # Load up the experiments for potential updates:
    experiments = pd.read_csv("experiments.csv")
    experiments.index = experiments.RUN.values
    
    # Bypass the experiment if it is already performed
    if experiments.Done[experiment_i] == 0 and experiments.In_Progress.loc[experiment_i] == 0:
        
        #### Log that the experiment is being processed (paralellism)
        if store_progress == True:
            experiments.In_Progress.loc[experiments.RUN == experiment_i] = 1
            experiments.to_csv("experiments.csv",index=False)
        
        #### Generate a folder for the experiment
        
        # importing os module 
        import os 
          
        #Experiment
        path = os.path.join(parent_dir, str(experiment_i)) 
        
        if not os.path.exists(path):
            os.mkdir(path) 
            
            #inference tables
            path2 = os.path.join(path, "inference_tables") 
            os.mkdir(path2) 
            
            #models
            path3 = os.path.join(path, "models") 
            os.mkdir(path3) 
            
            #models
            path4 = os.path.join(path, "individuals") 
            os.mkdir(path4) 
            
            #training logfiles 
            path5 = os.path.join(path, "train_logfiles") 
            os.mkdir(path5) 
        
        
        ###############################################################################
        # Store the experiment-settings for reference in the evaluation functions
        ###############################################################################
        
        # Get the number of the experiment
        RUN = experiments.RUN[experiment_i]
        
        # Load the levels of the experiment
        Experiment_Settings = np.load('Experiment_Settings.npy',allow_pickle='TRUE').item()
        
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
            
            #Load without conversion, since JMP encode with correct level values
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
        Finalmodel_epochs = 250


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
        
        print("================================"*3)
        print(configfile.loc[0])
        print("================================"*3)
        
        ###############################################################################
        # Automated model search approaches
        ###############################################################################
        
        if F_modelselection == "Multiple":
            # Store starttime
            start_time = time.time()
            
            # New toolbox instance with the necessary components.
            toolbox = base.Toolbox()
            
            creator.create('FitnessMax', base.Fitness, weights = (-1.0, -1.0))
            creator.create('Individual', list , fitness = creator.FitnessMax)
            
            toolbox = base.Toolbox()
            toolbox.register('binary', bernoulli.rvs, 0.5)
            toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary, n = gene_length)
            toolbox.register('population', tools.initRepeat, list , toolbox.individual)
            
            # Describing attributes, individuals and population and defining the selection, mating and mutation operators.
            
            # Binary mate, mutate 
            toolbox.register('mate', tools.cxOrdered)
            toolbox.register('mutate', tools.mutShuffleIndexes, indpb = 0.5) #50% chance of shuffling attr
            
            # NSGA-II selection
            toolbox.register("select", tools.selNSGA2)
            
            # Custom evaluation function
            toolbox.register("evaluate", train_evaluate)
            
            
            # Use the toolbox to store other configuration parameters of the algorithm. 
            toolbox.pop_size = population_size
            toolbox.max_gen = num_generations
            toolbox.mut_prob = F_mutation_prob
            
            pop = toolbox.population(n=toolbox.pop_size)
            pop = toolbox.select(pop, len(pop))
            
            
            # Save some statistics:
            statistics = tools.Statistics(key=lambda ind: ind.fitness.values)
            #statistics.register("avg", np.mean)
            statistics.register("avg", np.mean, axis=0)
            statistics.register("std", np.std, axis=0)
            statistics.register("min", np.min, axis=0)
            statistics.register("max", np.max, axis=0)
            
            # Save hall of fame:
            hof = tools.HallOfFame(k_in_hall_of_fame)
            
            
            # A compact NSGA-II implementation
            
            # Storing all the required information in the toolbox and using DEAP's 
            # algorithms.eaMuPlusLambda function allows us to create a very compact 
            # -albeit not a 100% exact copy of the original- implementation of NSGA-II.
            
            
            lastgen, logbook = algorithms.eaMuPlusLambda(pop, toolbox, 
                                                 mu=toolbox.pop_size, #The number of individuals to select for the next generation.
                                                 lambda_= toolbox.pop_size + elitism, #The number of children to produce at each generation.
                                                 cxpb=1-toolbox.mut_prob, #CXPB
                                                 mutpb=toolbox.mut_prob, #MUTPB
                                                 stats=statistics, 
                                                 halloffame=hof, #Save the all-time K best individuals
                                                 ngen=toolbox.max_gen, #The number of generation.
                                                 verbose=True)
            
            # CXPB is the probability that an offspring is produced by crossover.
            # MUTPB is The probability that an offspring is produced by mutation
            # INDPB is the probability of each attribute to be moved 
            
            ###############################################################################
            # Evaluate after search
            ###############################################################################
            
            #overwrite last gen with hall of fame top ensure best results
            HOF = hof.items
            
            # Store the used datasets for faster evaluation:
            data_objects = prepare_dataset(suffix=F_dataset, sample=1.0)
            
            hof_results = SaveLastGenResults(data_objects, HOF, logbook, RUN, start_time)
            
            SaveLogbookResults(logbook, F_modelselection, RUN, population_size, num_generations, gene_length)
            
            ###############################################################################
            # Pareto front
            ###############################################################################
            
            """
            The fronts are only derived from the last population,
            and each solution is therefore a non-dominated solution 
            from the last generation/population.
            
            For small populations, and few generations, fewer solutions
            might be non-dominated.
            """

            StoreParetoFronts(data_objects, lastgen, RUN)
        
            ###############################################################################
            
            # Store the amount of time the experiment took
            end_time = time.time()
            Time_sec = end_time - start_time
        
        if F_modelselection == "Single-MAE" or F_modelselection == "Single-MAEPE" or F_modelselection == "Single-MEP":
            # Store starttime
            start_time = time.time()
            
            # As we are trying to minimize the RMSE score, that's why using -1.0. 
            # In case, when you want to maximize accuracy for instance, use 1.0
            
            creator.create('FitnessMax', base.Fitness, weights = (-1.0,))
            creator.create('Individual', list , fitness = creator.FitnessMax)
            
            toolbox = base.Toolbox()
            toolbox.register('binary', bernoulli.rvs, 0.5)
            toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary, n = gene_length)
            toolbox.register('population', tools.initRepeat, list , toolbox.individual)
            
            toolbox.register('mate', tools.cxOrdered)
            toolbox.register('mutate', tools.mutShuffleIndexes, indpb = 0.5) #50% chance of shuffling attr
            toolbox.register('select', tools.selRoulette)
            toolbox.register('evaluate', train_evaluate)
            
            # Use the toolbox to store other configuration parameters of the algorithm. 
            
            toolbox.pop_size = population_size
            toolbox.max_gen = num_generations
            toolbox.mut_prob = F_mutation_prob
            
            pop = toolbox.population(n = toolbox.pop_size)
            
            # Save some statistics:
            statistics = tools.Statistics(key=lambda ind: ind.fitness.values)
            statistics.register("avg", np.mean)
            statistics.register("std", np.std)
            statistics.register("min", np.min)
            statistics.register("max", np.max)
            
            # Save hall of fame:
            hof = tools.HallOfFame(k_in_hall_of_fame)
            
            # A compact implementation
            
            # Storing all the required information in the toolbox and using DEAP's 
            # algorithms.eaMuPlusLambda function
            """
            res, logbook = algorithms.eaSimple(pop, toolbox, 
                                                 cxpb=1-toolbox.mut_prob, #CXPB
                                                 mutpb=toolbox.mut_prob, #MUTPB
                                                 stats=statistics, 
                                                 ngen=toolbox.max_gen, 
                                                 verbose=True)
            """
            lastgen, logbook = algorithms.eaMuPlusLambda(pop, toolbox, 
                                                 mu=toolbox.pop_size,       #The number of individuals to select for the next generation.
                                                 lambda_= toolbox.pop_size + elitism, #The number of children to produce at each generation.
                                                 cxpb=1-toolbox.mut_prob,   #CXPB
                                                 mutpb=toolbox.mut_prob,    #MUTPB
                                                 stats=statistics, 
                                                 halloffame=hof, #Save the all-time K best individuals
                                                 ngen=toolbox.max_gen, #The number of generation.
                                                 verbose=True)
            
            # CXPB is the probability that an offspring is produced by crossover.
            # MUTPB is The probability that an offspring is produced by mutation
            # INDPB is the probability of each attribute to be moved 
            
            ###############################################################################
            # Evaluate after search
            ###############################################################################
            
            #overwrite last gen with hall of fame top ensure best results
            HOF = hof.items
            
            # Store the used datasets for faster evaluation:
            data_objects = prepare_dataset(suffix=F_dataset, sample=1.0)
            
            hof_results = SaveLastGenResults(data_objects, HOF, logbook, RUN, start_time)
                        
            SaveLogbookResults(logbook, F_modelselection, RUN, population_size, num_generations, gene_length)
            
            ################################################################################
            # Store the amount of time the experiment took
            end_time = time.time()
            Time_sec = end_time - start_time
            
        if F_modelselection == "RS":
            # Store starttime
            start_time = time.time()
            
            # Determine number of model searches
            n_models =  population_size * (num_generations)
            
            list_of_models = []
            
            # Run n random search iterations
            for i in range(0,n_models):
                print("model ",str(i))
                
                #Generate a random individual
                individual = np.random.randint(2, size=(gene_length,))
                
                #Store it on the list of models
                list_of_models.append(individual)
                
                # Train a model based on the individual:
                ACC, ACCEAR, EAR = train_evaluate(individual)
                        
            ###############################################################################
            # Evaluate after search
            ###############################################################################
            
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
            #last_gen_results.to_csv("experiments/"+str(RUN)+"/"+str(RUN)+"_RS_all_models_results.csv",index=False)
            
            last_gen_results["Search"] = "RS"
            last_gen_results.to_csv("experiments/"+str(RUN)+"/HOF_results.csv",index=False)
            
            hof_results = last_gen_results
        
        ###############################################################################
        # POST model search
        ###############################################################################
        # Find the best model in the last generation
        
        experiments.MAE_min.loc[experiments.RUN == experiment_i] = float(np.min(hof_results["MAE"].values))
        experiments.MAE_max.loc[experiments.RUN == experiment_i] = float(np.max(hof_results["MAE"].values))
        experiments.MAE_avg.loc[experiments.RUN == experiment_i] = float(np.mean(hof_results["MAE"].values))
        experiments.MAE_std.loc[experiments.RUN == experiment_i] = float(np.std(hof_results["MAE"].values))
        
        experiments.MEP_min.loc[experiments.RUN == experiment_i] = float(np.min(hof_results["MEP"].values))
        experiments.MEP_max.loc[experiments.RUN == experiment_i] = float(np.max(hof_results["MEP"].values))
        experiments.MEP_avg.loc[experiments.RUN == experiment_i] = float(np.mean(hof_results["MEP"].values))
        experiments.MEP_std.loc[experiments.RUN == experiment_i] = float(np.std(hof_results["MEP"].values))
        
        experiments.MAEPE_min.loc[experiments.RUN == experiment_i] = float(np.min(hof_results["MAEPE"].values))
        experiments.MAEPE_max.loc[experiments.RUN == experiment_i] = float(np.max(hof_results["MAEPE"].values))
        experiments.MAEPE_avg.loc[experiments.RUN == experiment_i] = float(np.mean(hof_results["MAEPE"].values))
        experiments.MAEPE_std.loc[experiments.RUN == experiment_i] = float(np.std(hof_results["MAEPE"].values))
        
        experiments.Num_models.loc[experiments.RUN == experiment_i] = len(hof_results)
        
        # Log the status of the experiment
        print("================================"*3)
        print("Finished experiment",RUN)
        print("================================"*3)
        experiments.Done.loc[experiments.RUN == experiment_i] = 1
        experiments.In_Progress.loc[experiments.RUN == experiment_i] = 0
        experiments.Duration_sec.loc[experiments.RUN == experiment_i] = Time_sec
        experiments.to_csv("experiments.csv",index=False)
        
        
        
"""
#######################################################################################
"""



# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 18:03:46 2020

@author: Mike
"""


from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from bitstring import BitArray
import time

import numpy as np
import pandas as pd
from Model_search_helpers import *
from Eval_helpers import *

from HPO_searchspace import *

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
        
        print("================================"*3)
        print(configfile.loc[0])
        print("================================"*3)
        
        
        ###############################################################################
        # Load the best K models from the experiment
        ###############################################################################
        
        if F_modelselection == "RS":
            
            Results = pd.read_csv("experiments/"+str(RUN)+"/HOF_results.csv") #"+str(RUN)+"_
            Winner = Results.sort_values(by=['MAE'], ascending=True).reset_index(drop=True).loc[0]
                        
            # MAE example:
            individual = np.fromstring(np.binary_repr(Winner["individual"]), dtype='S1').astype(int)
            Experiment_results = train_evaluate(individual, train_final_model=True)
        
        ###############################################################################
        if F_modelselection == "Multiple":
            
            Results = pd.read_csv("experiments/"+str(RUN)+"/HOF_results.csv") #"+str(RUN)+"_
            
            # Selection according to MAE or MAEPE:
            Pareto_winner = Results.loc[0]
            
            # Pareto:
            individual = np.fromstring(np.binary_repr(Pareto_winner["individual"]), dtype='S1').astype(int)
            Experiment_results = train_evaluate(individual, train_final_model=True)
            
        
        ###############################################################################
        if F_modelselection == "Single-MAE" or F_modelselection == "Single-MEP":
            
            Results = pd.read_csv("experiments/"+str(RUN)+"/HOF_results.csv") #"+str(RUN)+"_
            
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
            Final_training_results.to_csv("experiments/Final_models/Final_model_training_results.csv",index=False)
                        
        if len(Final_training_results) == 0:
            Final_training_results = Experiment_results
            Final_training_results.to_csv("experiments/Final_models/Final_model_training_results.csv",index=False)
            
        ###############################################################################
        # Log the status of the experiment
        print("================================"*3)
        print("Finished experiment",RUN)
        print("================================"*3)
        experiments.Finalmodel_Done.loc[experiments.RUN == experiment_i] = 1
        experiments.to_csv("experiments.csv",index=False)
            
        
if RUN == Max_models:
    #Create the final table:
    Final_training_results = pd.read_csv("experiments/Final_models/Final_model_training_results.csv")
    experiments_merged = experiments.merge(Final_training_results, on='RUN', how='left')
    experiments_merged.to_csv("Final_experiments_merged.csv",index=False)

