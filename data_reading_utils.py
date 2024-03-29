import pandas as pd
import numpy as np
#=================================================================================================
# Physiological data read according to the client
#=================================================================================================

##===================================================
# EEG data read from files
##===================================================
# def eeg_data(p):
#     file_eeg = '/home/gp/Desktop/MER_arin/DEAP_data/eeg_data/'+str(p)+'_data_DEAP'+'.csv'
#     print(file_eeg)
#     eeg_sig = pd.read_csv(file_eeg,sep=',', header = None, engine='python')
#     return eeg_sig

##===================================================
# EDA data read from files
##===================================================
def eda_data(p):
    file_eda = '/home/gp/Desktop/PhD-codes/DEAP_data/eda_data/'+str(p)+'_GSR_data_from_DEAP.csv'
    print(file_eda)
    eda_sig = pd.read_csv(file_eda,sep=',', header = None, engine='python')
    return eda_sig

##===================================================
# Resp data read from files
##===================================================
def resp_data(p):
    file_resp = '/home/gp/Desktop/PhD-codes/DEAP_data/resp_data/'+str(p)+'_Respiration_data_from_DEAP.csv'
    print(file_resp)
    resp_sig = pd.read_csv(file_resp,sep=',', header = None, engine='python')
    return resp_sig
