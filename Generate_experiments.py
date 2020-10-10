from helperfunctions import *
import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt

DOE = "Full_factorial"#"Fractional_factorial"

if DOE == "Full_factorial":
    #Simple setup for testing purposes:
    Settings = {'F_modeltype':["LSTM"],
                'F_modelselection':["Single-MAE"], #"RS","Single-MAE","Multiple", ,"Single-MEP"
                'F_dataset':["Helpdesk"],  # Does this differ over datasets? ,
                              #"Sepsis",
                              #"Hospital_billing",
                              #"SF_eventlog_filter_length1",
                              #"traffic_fines"
                'F_mutation_prob':[0.02], #0.02, 0.1, 0.2, 0.5 #Same as in GA paper (Exploitation), balanced, and exploration (lots of randomness)
                'F_num_generations':[3], #2, 10
                'F_population_size':[5], #5, 20
                'F_lofi_epochs':[150]} #1, 10, 20, 30, 40, 50 #,10 # Fast, balanced, thorough
        
    
    # Generate a full factorial:
    df=build_full_fact(Settings)  
    
    
    # Save the settings to a file
    np.save('Experiment_Settings.npy', Settings) 

if DOE == "Fractional_factorial":
    df = pd.read_csv("../FF_Experiments.csv")

#Generate a column for logging the status of the experiment (to pick up if it was interrupted)


# Constants affecting the experiments:
df["Finalmodel_Done"] = 0 #Flag if the final model has been trained
df["Duration_sec"] = 0

df["MAE_min"] = 0.0
df["MAE_max"] = 0.0
df["MAE_avg"] = 0.0
df["MAE_std"] = 0.0

df["MEP_min"] = 0.0
df["MEP_max"] = 0.0
df["MEP_avg"] = 0.0
df["MEP_std"] = 0.0

df["MAEPE_min"] = 0.0
df["MAEPE_max"] = 0.0
df["MAEPE_avg"] = 0.0
df["MAEPE_std"] = 0.0

df["Num_models"] = 0

df["RUN"] = df.index + 1
df["Done"] = 0
df["In_Progress"] = 0

print(df)

#store the new experimental design
df.to_csv("../Experiments.csv", index=False)