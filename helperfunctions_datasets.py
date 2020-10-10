# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 10:04:01 2019

@author: mikeriess


def InitialFormatting(df, maxcases, dateformat):
    import pandas as pd
    #Work on a subset:
    casestoload = df["id"].unique().tolist()[0:maxcases]
    df = df.loc[df["id"].isin(casestoload)]
    
  
    #Sort the dataframe by time aftewards
    df['parsed_date'] = pd.to_datetime(df.time, format = dateformat, exact = True)
    
    print("Sorting by id, date (chronological order):")
    #generate new ID column:
    df = df.assign(id=(df['id']).astype('category').cat.codes)
    #df = df.astype('str')
    df["id"] = df["id"].astype('int32') +1 
    df = df.sort_values(['id',"parsed_date"], ascending=[True, True]) # <----------------- here we can reverse the order of events
    
    
    df = df.drop("parsed_date",axis=1)
    return df

"""

import pandas as pd
import PPM_AUTO_EVAL.helperfunctions as hf


def GetFileInfo(df):
    print("Number of cases in log:",len(df["id"].unique()))
    import numpy as np
    import pandas as pd
    
    #Get the maximal trace length, for determining prefix length
    max_length = np.max(df['id'].value_counts())
    print("longest trace is:",max_length)
    
    #Look at the time format:
    print("Time format:",df["time"].loc[0])
    print("Std. format: %Y-%m-%d %H:%M:%S")
    
    print(df.head())
    return max_length

def LoadDataSet(file, No_features, Max_num_cases):
    import pandas as pd
    import numpy as np
    import PPM_AUTO_EVAL.helperfunctions as hf
    
    if file == "Sepsis":
        separator=";"
        dateformat = "%Y/%m/%d %H:%M:%S.%f"
        colmapping = {'Case ID':'id',
                      'Activity':'event',
                      'Complete Timestamp':'time'}
        df = pd.read_csv(str("data/"+file+".csv"),sep=separator)
        print("=======================================")
        print(df.columns.tolist())
        print("=======================================")
        #Rename vars to standard format to fit the rest of the code:
        df = df.rename(columns=colmapping, inplace=False)
        if No_features == False:
            #Categorical features to use:
            categoricals = ['Diagnose', 'DiagnosticArtAstrup', 'DiagnosticBlood', 'DiagnosticECG', 
                            'DiagnosticIC', 'DiagnosticLacticAcid', 'DiagnosticLiquor', 'DiagnosticOther', 
                            'DiagnosticSputum', 'DiagnosticUrinaryCulture', 'DiagnosticUrinarySediment', 
                            'DiagnosticXthorax', 'DisfuncOrg', 'Hypotensie', 'Hypoxie', 'InfectionSuspected', 
                            'Infusion', 'Oligurie', 'SIRSCritHeartRate', 'SIRSCritLeucos', 'SIRSCritTachypnea', 
                            'SIRSCritTemperature', 'SIRSCriteria2OrMore','org:group']
            #Numerical features to use:
            numericals = ['Age','CRP', 'LacticAcid', 'Leucocytes']#
        if No_features == True:    
            categoricals=[]
            numericals=[]
        #common preprocessing
        df = hf.InitialFormatting(df, maxcases = Max_num_cases,dateformat=dateformat)
        
    if file == "traffic_fines":
        separator=";"
        dateformat = "%Y-%m-%d %H:%M:%S"
        colmapping = {'Case ID':'id',
                      'Activity':'event',
                      'Complete Timestamp':'time'}
        df = pd.read_csv(str("data/"+file+".csv"),sep=separator)
        print("=======================================")
        print(df.columns.tolist())
        print("=======================================")
        #Rename vars to standard format to fit the rest of the code:
        df = df.rename(columns=colmapping, inplace=False)
        if No_features == False:
            #Categorical features to use:
            categoricals = ['article', 'vehicleClass','Resource', 'lastSent', 'notificationType', 'dismissal','label']
            #Numerical features to use:
            numericals = ['amount', 'points','expense','open_cases']#
        if No_features == True:    
            categoricals=[]
            numericals=[]
        #common preprocessing
        print("processing")
        df = hf.InitialFormatting(df, maxcases = Max_num_cases,dateformat=dateformat)
    
    
    if file == "SF_eventlog_filter_length1":
        separator="," 
        dateformat = "%Y-%m-%dT%H:%M:%S.%fZ" #2016-10-05T13:54:14.000Z
        colmapping = {'case_id':'id',
                      'task_tasksubtype':'event',
                      'task_createddate':'time'}
        df = pd.read_csv(str("data/"+file+".csv"), sep=separator)
        print("=======================================")
        print(df.columns.tolist())
        print("=======================================")
        #Rename vars to standard format to fit the rest of the code:
        df = df.rename(columns=colmapping, inplace=False)
        if No_features == False:
            #Categorical features to use:
            categoricals = ["ressource","ressource_role","case_owner_role","case_topic"]
            #Numerical features to use:
            numericals = ["task_number"]#
        if No_features == True:    
            categoricals=[]
            numericals=[]
        #common preprocessing
        print("processing")
        df = hf.InitialFormatting(df, maxcases = Max_num_cases, dateformat=dateformat)    
    
    
    if file == "sepsis_cases_3":
        separator=";"
        dateformat = "%Y-%m-%d %H:%M:%S"
        colmapping = {'Case ID':'id',
                      'Activity':'event',
                      'time:timestamp':'time'}
        df = pd.read_csv(str("data/"+file+".csv"),sep=separator)
        print("=======================================")
        print(df.columns.tolist())
        print("=======================================")
        #Rename vars to standard format to fit the rest of the code:
        df = df.rename(columns=colmapping, inplace=False)
        if No_features == False:
            #Categorical features to use:
            categoricals = ['Diagnose', 'DiagnosticArtAstrup', 'DiagnosticBlood', 'DiagnosticECG', 
                            'DiagnosticIC', 'DiagnosticLacticAcid', 'DiagnosticLiquor', 'DiagnosticOther', 
                            'DiagnosticSputum', 'DiagnosticUrinaryCulture', 'DiagnosticUrinarySediment', 
                            'DiagnosticXthorax', 'DisfuncOrg', 'Hypotensie', 'Hypoxie', 'InfectionSuspected', 
                            'Infusion', 'Oligurie', 'SIRSCritHeartRate', 'SIRSCritLeucos', 'SIRSCritTachypnea', 
                            'SIRSCritTemperature', 'SIRSCriteria2OrMore','org:group','open_cases', 'label']
            #Numerical features to use:
            numericals = ['Age','CRP', 'LacticAcid', 'Leucocytes']#
        if No_features == True:    
            categoricals=[]
            numericals=[]
        #common preprocessing
        df = hf.InitialFormatting(df, maxcases = Max_num_cases,dateformat=dateformat)
    
    
    
    if file == "sepsis_cases_2":
        separator=";"
        dateformat = "%Y-%m-%d %H:%M:%S"
        colmapping = {'Case ID':'id',
                      'Activity':'event',
                      'time:timestamp':'time'}
        df = pd.read_csv(str("data/"+file+".csv"),sep=separator)
        print("=======================================")
        print(df.columns.tolist())
        print("=======================================")
        #Rename vars to standard format to fit the rest of the code:
        df = df.rename(columns=colmapping, inplace=False)
        if No_features == False:
            #Categorical features to use:
            categoricals = ['Diagnose', 'DiagnosticArtAstrup', 'DiagnosticBlood', 'DiagnosticECG', 
                            'DiagnosticIC', 'DiagnosticLacticAcid', 'DiagnosticLiquor', 'DiagnosticOther', 
                            'DiagnosticSputum', 'DiagnosticUrinaryCulture', 'DiagnosticUrinarySediment', 
                            'DiagnosticXthorax', 'DisfuncOrg', 'Hypotensie', 'Hypoxie', 'InfectionSuspected', 
                            'Infusion', 'Oligurie', 'SIRSCritHeartRate', 'SIRSCritLeucos', 'SIRSCritTachypnea', 
                            'SIRSCritTemperature', 'SIRSCriteria2OrMore','org:group','open_cases', 'label']
            #Numerical features to use:
            numericals = ['Age','CRP', 'LacticAcid', 'Leucocytes']#
        if No_features == True:    
            categoricals=[]
            numericals=[]
        #common preprocessing
        df = hf.InitialFormatting(df, maxcases = Max_num_cases,dateformat=dateformat)
    
    
    if file == "sepsis_cases_1":
        separator=";"
        dateformat = "%Y-%m-%d %H:%M:%S"
        colmapping = {'Case ID':'id',
                      'Activity':'event',
                      'time:timestamp':'time'}
        df = pd.read_csv(str("data/"+file+".csv"),sep=separator)
        print("=======================================")
        print(df.columns.tolist())
        print("=======================================")
        #Rename vars to standard format to fit the rest of the code:
        df = df.rename(columns=colmapping, inplace=False)
        if No_features == False:
            #Categorical features to use:
            categoricals = ['Diagnose', 'DiagnosticArtAstrup', 'DiagnosticBlood', 'DiagnosticECG', 
                            'DiagnosticIC', 'DiagnosticLacticAcid', 'DiagnosticLiquor', 'DiagnosticOther', 
                            'DiagnosticSputum', 'DiagnosticUrinaryCulture', 'DiagnosticUrinarySediment', 
                            'DiagnosticXthorax', 'DisfuncOrg', 'Hypotensie', 'Hypoxie', 'InfectionSuspected', 
                            'Infusion', 'Oligurie', 'SIRSCritHeartRate', 'SIRSCritLeucos', 'SIRSCritTachypnea', 
                            'SIRSCritTemperature', 'SIRSCriteria2OrMore','org:group','open_cases', 'label']
            #Numerical features to use:
            numericals = ['Age','CRP', 'LacticAcid', 'Leucocytes']#
        if No_features == True:    
            categoricals=[]
            numericals=[]
        #common preprocessing
        df = hf.InitialFormatting(df, maxcases = Max_num_cases,dateformat=dateformat)
        
    
    if file == "hospital_billing":
        separator=";"
        dateformat = "%Y-%m-%d %H:%M:%S"
        colmapping = {'Case ID':'id',
                      'Activity':'event',
                      'Complete Timestamp':'time'}
        df = pd.read_csv(str("data/"+file+".csv"),sep=separator)
        print("=======================================")
        print(df.columns.tolist())
        print("=======================================")
        #Rename vars to standard format to fit the rest of the code:
        df = df.rename(columns=colmapping, inplace=False)
        if No_features == False:
            #Categorical features to use:
            categoricals = ['speciality','Resource', 'actOrange', 'actRed', 'blocked', 'caseType', 'diagnosis', 'flagC',
                            'flagD', 'msgCode', 'msgType', 'state', 'version', 'isCancelled', 'isClosed', 'closeCode',
                            'label']
            #Numerical features to use:
            numericals = ['msgCount','open_cases']#
        if No_features == True:    
            categoricals=[]
            numericals=[]
        #common preprocessing
        df = hf.InitialFormatting(df, maxcases = Max_num_cases,dateformat=dateformat)
        
    
    if file == "BPIC17_O_Refused":
        separator=";"
        dateformat = "%Y-%m-%d %H:%M:%S.%f"
        colmapping = {'Case ID':'id',
                      'Activity':'event',
                      'time:timestamp':'time'}
        df = pd.read_csv(str("data/"+file+".csv"),sep=separator)
        print("=======================================")
        print(df.columns.tolist())
        print("=======================================")
        #Rename vars to standard format to fit the rest of the code:
        df = df.rename(columns=colmapping, inplace=False)
        if No_features == False:
            #Categorical features to use:
            categoricals = ['ApplicationType', 'LoanGoal','label','org:resource', 'Action',
                            'EventOrigin', 'lifecycle:transition', 'Accepted',
                            'Selected']
            #Numerical features to use:
            numericals = ['RequestedAmount','FirstWithdrawalAmount', 'MonthlyCost', 
                          'NumberOfTerms', 'OfferedAmount', 'CreditScore', 'open_cases']#
        if No_features == True:    
            categoricals=[]
            numericals=[]
        #common preprocessing
        df = hf.InitialFormatting(df, maxcases = Max_num_cases,dateformat=dateformat)
        
    
    if file == "BPIC17_O_Cancelled":
        separator=";"
        dateformat = "%Y-%m-%d %H:%M:%S.%f"
        colmapping = {'Case ID':'id',
                      'Activity':'event',
                      'time:timestamp':'time'}
        df = pd.read_csv(str("data/"+file+".csv"),sep=separator)
        print("=======================================")
        print(df.columns.tolist())
        print("=======================================")
        #Rename vars to standard format to fit the rest of the code:
        df = df.rename(columns=colmapping, inplace=False)
        if No_features == False:
            #Categorical features to use:
            categoricals = ['ApplicationType', 'LoanGoal','label','org:resource', 'Action',
                            'EventOrigin', 'lifecycle:transition', 'Accepted',
                            'Selected']
            #Numerical features to use:
            numericals = ['RequestedAmount','FirstWithdrawalAmount', 'MonthlyCost', 
                          'NumberOfTerms', 'OfferedAmount', 'CreditScore', 'open_cases']#
        if No_features == True:    
            categoricals=[]
            numericals=[]
        #common preprocessing
        df = hf.InitialFormatting(df, maxcases = Max_num_cases,dateformat=dateformat)
    
    
    if file == "BPIC17_O_Accepted":
        separator=";"
        dateformat = "%Y-%m-%d %H:%M:%S.%f"
        colmapping = {'Case ID':'id',
                      'Activity':'event',
                      'time:timestamp':'time'}
        df = pd.read_csv(str("data/"+file+".csv"),sep=separator)
        print("=======================================")
        print(df.columns.tolist())
        print("=======================================")
        #Rename vars to standard format to fit the rest of the code:
        df = df.rename(columns=colmapping, inplace=False)
        if No_features == False:
            #Categorical features to use:
            categoricals = ['ApplicationType', 'LoanGoal','label','org:resource','Action',
                            'EventOrigin', 'lifecycle:transition', 'Accepted',
                            'Selected']
            #Numerical features to use:
            numericals = ['RequestedAmount','FirstWithdrawalAmount', 'MonthlyCost', 
                          'NumberOfTerms', 'OfferedAmount', 'CreditScore', 'open_cases']#
        if No_features == True:    
            categoricals=[]
            numericals=[]
        #common preprocessing
        df = hf.InitialFormatting(df, maxcases = Max_num_cases,dateformat=dateformat)
        
    
    if file == "bpic2012_O_ACCEPTED-COMPLETE":
        separator=";"
        dateformat = "%Y-%m-%d %H:%M:%S.%f"
        colmapping = {'Case ID':'id',
                      'Activity':'event',
                      'Complete Timestamp':'time'}
        df = pd.read_csv(str("data/"+file+".csv"),sep=separator)
        print("=======================================")
        print(df.columns.tolist())
        print("=======================================")
        #Rename vars to standard format to fit the rest of the code:
        df = df.rename(columns=colmapping, inplace=False)
        if No_features == False:
            #Categorical features to use:
            categoricals = ["Resource","label",'lifecycle:transition','open_cases']
            #Numerical features to use:
            numericals = ["AMOUNT_REQ"]#
        if No_features == True:    
            categoricals=[]
            numericals=[]
        #common preprocessing
        df = hf.InitialFormatting(df, maxcases = Max_num_cases,dateformat=dateformat)
        
        
    if file == "bpic2012_O_CANCELLED-COMPLETE":
        separator=";"
        dateformat = "%Y-%m-%d %H:%M:%S.%f"
        colmapping = {'Case ID':'id',
                      'Activity':'event',
                      'Complete Timestamp':'time'}
        df = pd.read_csv(str("data/"+file+".csv"),sep=separator)
        print("=======================================")
        print(df.columns.tolist())
        print("=======================================")
        #Rename vars to standard format to fit the rest of the code:
        df = df.rename(columns=colmapping, inplace=False)
        if No_features == False:
            #Categorical features to use:
            categoricals = ["Resource","label",'lifecycle:transition','open_cases']
            #Numerical features to use:
            numericals = ["AMOUNT_REQ"]#
        if No_features == True:    
            categoricals=[]
            numericals=[]
        #common preprocessing
        df = hf.InitialFormatting(df, maxcases = Max_num_cases,dateformat=dateformat)
        
        
    if file == "bpic2012_O_DECLINED-COMPLETE":
        separator=";"
        dateformat = "%Y-%m-%d %H:%M:%S.%f"
        colmapping = {'Case ID':'id',
                      'Activity':'event',
                      'Complete Timestamp':'time'}
        df = pd.read_csv(str("data/"+file+".csv"),sep=separator)
        print("=======================================")
        print(df.columns.tolist())
        print("=======================================")
        #Rename vars to standard format to fit the rest of the code:
        df = df.rename(columns=colmapping, inplace=False)
        if No_features == False:
            #Categorical features to use:
            categoricals = ["Resource","label",'lifecycle:transition','open_cases']
            #Numerical features to use:
            numericals = ["AMOUNT_REQ"]#
        if No_features == True:    
            categoricals=[]
            numericals=[]
        #common preprocessing
        df = hf.InitialFormatting(df, maxcases = Max_num_cases,dateformat=dateformat)
    
    
    if file == "Helpdesk":#2017_rearranged
        separator=","
        dateformat = "%Y-%m-%d %H:%M:%S"
        colmapping = {'Case ID':'id',
                      'Activity':'event',
                      'Complete Timestamp_formatted':'time'}
        df = pd.read_csv(str("data/"+file+".csv"),sep=separator)
        print("=======================================")
        print(df.columns.tolist())
        print("=======================================")
        #Rename vars to standard format to fit the rest of the code:
        df = df.rename(columns=colmapping, inplace=False)
        if No_features == False:
            #Categorical features to use:
            categoricals = ["Resource","seriousness","product",
                            "responsible_section","seriousness_2","service_level", #,"Variant_2"
                            "service_type","support_section","workgroup"]
            #Numerical features to use:
            numericals = []#
        if No_features == True:    
            categoricals=[]
            numericals=[]
        #common preprocessing
        df = hf.InitialFormatting(df, maxcases = Max_num_cases,dateformat=dateformat)
        
    if file == "Testdata":
        separator=";"
        dateformat = "%Y-%m-%d %H:%M:%S"
        colmapping = {'Case ID':'id',
                      'Activity':'event',
                      'Complete Timestamp_formatted':'time'}
        df = pd.read_csv(str("data/"+file+".csv"),sep=separator)
        print("=======================================")
        print(df.columns.tolist())
        print("=======================================")
        #Rename vars to standard format to fit the rest of the code:
        df = df.rename(columns=colmapping, inplace=False)
        if No_features == False:
            #Categorical features to use:
            categoricals = ["Cat_test"]
            #Numerical features to use:
            numericals = ["Num_test"]#
        if No_features == True:    
            categoricals=[]
            numericals=[]
        #common preprocessing
        df = hf.InitialFormatting(df, maxcases = Max_num_cases,dateformat=dateformat)
    
    return df, categoricals, numericals, dateformat