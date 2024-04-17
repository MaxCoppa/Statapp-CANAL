import sys
sys.path.append("Data_operations")

from Tool_Functions.cleaning_data import *
from sklearn.preprocessing import StandardScaler 

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


from preparation_data_set import * 
from new_data_set import * 
from viualize_datas import * 
from new_data_set_all import *

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from openpyxl import load_workbook

from matplotlib.backends.backend_pdf import PdfPages


data_path = "/Users/maximecoppa/Desktop/Statapp_Data/Datas/"

def test_data_set():
    id_abonne = np.array([0,1,2,3,4])
    nb_odd_15_tc = np.array([5,6,7,0,1])
    nb_odd_30_tc = np.array([7,8,9,1,0])



    df = pd.DataFrame({
        'ID_ABONNE': id_abonne,
        'NB_ODD_15_TC': nb_odd_15_tc,
        'NB_ODD_30_TC': nb_odd_30_tc,
    })

    return df


def cluster_data_set(filename,columns,change_inf = np.nan, change_nan = 15 ):
    
    df = file_to_dataframe(filename)
    data = df[columns]
    data.replace([np.inf, -np.inf], change_inf, inplace=True)

    scaler = StandardScaler()
    datas = scaler.fit_transform(data)

    data = pd.DataFrame(datas,columns = data.columns)

    indices = np.random.choice(range(len(data)), size=int(len(data) * 0.1), replace=False)

    data = data.iloc[indices]
    data.replace(np.nan,change_nan, inplace=True)

    return data



"""
df = file_to_dataframe(data_path + "data_clustering.csv")
print(df.columns)
data = df[['ODD 15 jours TC_MEAN_TIME_DIFF']]
data.replace([np.inf, -np.inf], np.nan, inplace=True)

scaler = StandardScaler()
datas = scaler.fit_transform(data)

print(datas)



data = pd.DataFrame(datas,columns = data.columns)

indices = np.random.choice(range(len(data)), size=int(len(data) * 0.1), replace=False)
print(indices)
data = data.iloc[indices]
data['ODD 15 jours TC_MEAN_TIME_DIFF'].replace(np.nan,15, inplace=True)

clusterer = KMeans(2, random_state=10)
clusterer.fit(data)
centers = clusterer.cluster_centers_
print(centers)
print(scaler.inverse_transform(centers))
"""






"""
kmeans = KMeans(n_clusters=2, random_state=10)
kmeans.fit(data)
print(kmeans.labels_)
l = kmeans.labels_
print(l.sum())
score = silhouette_score(data, kmeans.labels_)

print(score)

print(data.columns)
clusterer = KMeans(2, random_state=10)
print(clusterer)
cluster_labels = clusterer.fit_predict(datas)
print(cluster_labels)
silhouette_avg = silhouette_score(datas, cluster_labels)

print(silhouette_avg)

kmeans = KMeans(n_clusters=2, random_state=10)
kmeans.fit(datas)
score = silhouette_score(datas, kmeans.labels_)

print(score)

"""

def data_frame_cluster(data, columns , centers_inv, clusters, data_id_abo):

    data['KMEANS'] = clusters
    data['ID_ABONNE'] = data_id_abo['ID_ABONNE']

    df_clusters = percent_abo_conditions(data,'KMEANS','ID_ABONNE')
    df_clusters = df_clusters.sort_values(by = 'KMEANS')

    centers = np.round(centers_inv,decimals=2)

    for j in range(len(columns)) : 
        df_clusters[columns[j]] = [centers[i][j] for i in range(len(centers))]

    
    return df_clusters


def visualize_silhouette_datas(data, range_n_clusters, output_filename):


    
    with PdfPages(output_filename) as pdf:
        for n_clusters in range_n_clusters:

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters, random_state=10)
            clusterer.fit(data)

            centers = clusterer.cluster_centers_
            centers_cluster = np.round(scaler.inverse_transform(centers),decimals=2)
            
            print(centers_cluster)
            
            cluster_labels = clusterer.fit_predict(data)

            silhouette_avg = silhouette_score(data, cluster_labels)
            print(
                "For n_clusters =", n_clusters,
                "The average silhouette_score is :", silhouette_avg,
            )

            fig, ax1 = plt.subplots(1, 1)  # Modification ici : un seul subplot au lieu de deux
            fig.set_size_inches(9, 7)

            # Le reste du code reste inchangé
            # Veuillez ajouter votre code pour tracer les graphiques ici

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])

            y_lower = 10
            sample_silhouette_values = silhouette_samples(data, cluster_labels)

            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.8, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters." + str(n_clusters))
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

            pdf.savefig(fig)  # Sauvegarder la figure dans le document PDF
            plt.close(fig)  # Fermer la figure pour libérer la mémoire

    return True

def visualize_silhouette_datas_all(filename, columns , range_n_clusters, output_filename, change_inf = np.nan, change_nan = 15):

       
    df = file_to_dataframe(filename)
    data = df[columns]
    data.replace([np.inf, -np.inf], change_inf, inplace=True)

    scaler = StandardScaler()
    datas = scaler.fit_transform(data)

    data = pd.DataFrame(datas,columns = data.columns)

    indices = np.random.choice(range(len(data)), size=int(len(data) * 0.1), replace=False)

    data = data.iloc[indices]
    data_id_abo = df[['ID_ABONNE']].iloc[indices]
    data.replace(np.nan,change_nan, inplace=True)

    silhouette_scores = []
    
    with PdfPages(output_filename) as pdf:
        for n_clusters in range_n_clusters:

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters, random_state=10)
            clusterer.fit(data)

            centers = clusterer.cluster_centers_
            centers_cluster = np.round(scaler.inverse_transform(centers),decimals=2)
            
            
            df_cluster = data_frame_cluster(data,columns, scaler.inverse_transform(centers),clusterer.labels_, data_id_abo)
            write_df_to_excel(df_cluster,data_path + "cluster" + str(n_clusters) + ".xlsx",str(n_clusters))
            
            cluster_labels = clusterer.fit_predict(data)

            silhouette_avg = silhouette_score(data, cluster_labels)

            silhouette_scores.append(silhouette_avg)

            print(
                "For n_clusters =", n_clusters,
                "The average silhouette_score is :", silhouette_avg,
            )

            fig, ax1 = plt.subplots(1, 1)  # Modification ici : un seul subplot au lieu de deux
            fig.set_size_inches(9, 7)

            # Le reste du code reste inchangé
            # Veuillez ajouter votre code pour tracer les graphiques ici

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])

            y_lower = 10
            sample_silhouette_values = silhouette_samples(data, cluster_labels)

            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.8, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters." + str(n_clusters))
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

            pdf.savefig(fig)  # Sauvegarder la figure dans le document PDF
            plt.close(fig)  # Fermer la figure pour libérer la mémoire

    return silhouette_scores

def trace_silouhette_scores(silhouette_scores, abscisses):

    plt.figure(figsize=(8, 6))
    plt.plot(abscisses, silhouette_scores, marker='o')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Score de silhouette KMeans')
    plt.title('Score de silhouette pour différents nombres de clusters')
    plt.show()

    plt.xticks(abscisses)

    return True

def write_df_to_excel(df, file_name, sheet_name='Sheet1'):
    """
    Écrit un DataFrame dans un fichier Excel en spécifiant le nom du classeur.

    Args:
    - df: DataFrame pandas à écrire dans le fichier Excel.
    - file_name: Nom du fichier Excel.
    - sheet_name: Nom du classeur dans le fichier Excel (par défaut 'Sheet1').
    """
    # Création d'un writer avec le nom de fichier spécifié
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')

    # Écriture du DataFrame dans le classeur spécifié
    df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Sauvegarde et fermeture du fichier Excel
    writer.save()


    return True 


#data = cluster_data_set(data_path + "data_clustering.csv",['ODD 15 jours TC_MEAN_TIME_DIFF'])
trace_silouhette_scores(visualize_silhouette_datas_all(data_path + "data_clustering.csv",['ODD 15 jours TC_MEAN_TIME_DIFF'],[2,3,4],data_path + "test6.pdf"),[2,3,4])
#visualize_silhouette_datas(df[['NB_ODD_15_TC','NB_ODD_30_TC']],[2, 3, 4])

"""
columns = ['ODD 15 jours TC_MEAN_TIME_DIFF','ODD 15 jours TC_n_REABOS'] 
df = file_to_dataframe(data_path + "data_clustering.csv")
data = df[columns]
data.replace([np.inf, -np.inf], np.nan, inplace=True)

scaler = StandardScaler()
datas = scaler.fit_transform(data)

print(scaler.mean_)
print(scaler.var_)

def data_frame_cluster(data, columns , centers_inv, clusters, data_id_abo):

    data['KMEANS'] = clusters
    data['ID_ABONNE'] = data_id_abo['ID_ABONNE']

    df_clusters = percent_abo_conditions(data,'KMEANS','ID_ABONNE')
    df_clusters = df_clusters.sort_values(by = 'KMEANS')

    centers = np.round(centers_inv,decimals=2)

    for j in range(len(columns)) : 
        df_clusters[columns[j]] = [centers[i][j] for i in range(len(centers))]

    
    return df_clusters

data = pd.DataFrame(datas,columns = data.columns)

indices = np.random.choice(range(len(data)), size=int(len(data) * 0.1), replace=False)

data = data.iloc[indices]
data_id_abo = df[['ID_ABONNE']].iloc[indices]
data['ODD 15 jours TC_MEAN_TIME_DIFF'].replace(np.nan,15, inplace=True)

clusterer = KMeans(3, random_state=10)
clusterer.fit(data)
centers = clusterer.cluster_centers_


print(centers)

data['KMEANS'] = clusterer.labels_
data['ID_ABONNE'] = data_id_abo['ID_ABONNE']


df_clusters = percent_abo_conditions(data,'KMEANS','ID_ABONNE')
df_clusters = df_clusters.sort_values(by = 'KMEANS')



centers = np.round(scaler.inverse_transform(centers),decimals=2)

for j in range(len(columns)) : 
    df_clusters[columns[j]] = [centers[i][j] for i in range(len(centers))]

print(centers)
print(df_clusters)



print(data_frame_cluster(data,columns, scaler.inverse_transform(centers),clusterer.labels_, data_id_abo))

write_df_to_excel(data_frame_cluster(data,columns, scaler.inverse_transform(centers),clusterer.labels_, data_id_abo),data_path + "test.xlsx")

"""

