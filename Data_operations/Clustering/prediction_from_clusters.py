import sys
sys.path.append("Data_operations")

from Tool_Functions.cleaning_data import * 

from preparation_data_set import * 
from new_data_set import * 
from viualize_datas import * 
from new_data_set_all import *

data_path = "/Users/maximecoppa/Desktop/Statapp_Data/Datas/"


df = file_to_dataframe(data_path + "data_prediction.csv")

print(df.columns)

print(percent_abo_conditions(df,['Cluster_8','ANCIENNETE'],'ID_ABONNE'))