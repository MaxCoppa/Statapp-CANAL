import pandas as pd
import math
import sys 
sys.path.append("Data_operations")

from Tool_Functions.test_comportment_reabo import *

data_path = "/Users/clementgadeau/Statapp/CSV files/" #where to find the datas used
data_path_results = "/Users/clementgadeau/Statapp/StatDescr/" #where to create your new file

#creation_df_odd(data_path, data_path)
#create_df_Données_Promos_odd(data_path, data_path)
repartition_reabo_cond(data_path, data_path_results)
repartition_reabo(data_path, data_path_results)