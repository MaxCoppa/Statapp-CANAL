import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
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


# Silhouette score
def silhouette_scores_at_each_step(X, n_clusters_range):
    silhouette_scores = []
    for n_clusters in n_clusters_range:
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', metric='euclidean')
        labels = clustering.fit_predict(X)
        if len(np.unique(labels)) == 1:
            silhouette_scores.append(-1)  # Silhouette score not computable
        else:
            silhouette_scores.append(silhouette_score(X, labels))
    return silhouette_scores

n_clusters_range = range(data.shape[0], data.shape[0] +1, 1)
silhouette_scores = silhouette_scores_at_each_step(data, n_clusters_range)

# Plot
plt.plot(n_clusters_range, silhouette_scores, marker='o')
plt.xlabel('Nombre de clusters')
plt.ylabel('Score de silhouette')
plt.title('Score de silhouette à chaque étape')
plt.show()