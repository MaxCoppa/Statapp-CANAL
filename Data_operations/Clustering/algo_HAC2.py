import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import s3fs

# Create filesystem object
S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})
FILE_KEY_S3 = "csv files/fusion_table_v3.csv"
FILE_PATH_S3 = "cgadeau/" + FILE_KEY_S3

with fs.open(FILE_PATH_S3, mode="rb") as file_in:
    df_bpe = pd.read_csv(file_in, sep=",")

# Clean
data = df_bpe
data = data.drop(columns = 'NOMBRE_ABONNEMENTS')
data = data.astype(float)
data.replace(np.inf, 1120, inplace = True)

# Standardize
scaler = StandardScaler()
for column in data.columns.tolist():
    data[column]  = scaler.fit_transform(data[column].values.reshape(-1, 1))


clustering = AgglomerativeClustering(n_clusters=data.shape[0]-1, linkage='ward', affinity='euclidean')
clustering.fit(data)
print(clustering.labels_)