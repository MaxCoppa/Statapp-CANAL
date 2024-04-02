import sys
sys.path.append("Data_operations")

from Tool_Functions.cleaning_data import *
from Tool_Functions.comportment_reabo import * 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt
import math 
import numpy as np

import matplotlib.cm as cm

from matplotlib.backends.backend_pdf import PdfPages


 

def normalise_column(df,name_colum,x = -3):

    df[name_colum] = df[name_colum].replace([np.inf, -np.inf], np.nan)

    column_mean = df[name_colum].mean()
    sqrt_var = np.sqrt(df[name_colum].var())

    df[name_colum] = (df[name_colum] - column_mean)/sqrt_var

    df[name_colum] = df[name_colum].replace(np.nan,x)

    return df

def normalise_data_frame(df,columns):

    for el in columns : 

        df = normalise_column(df,el)
   
    return df[columns]


def score_silouhette(df,columns,m = 2, n = 11):

    df_scaled = normalise_data_frame(df,columns)

    np.random.seed(42)
    indices = np.random.choice(range(len(df_scaled)), size=int(len(df_scaled) * 0.1), replace=False)
    sample = df_scaled.iloc[indices]
    silhouette_scores = []

    for k in range(m, n):  # Testez des valeurs de k de 2 à 10
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(sample)
        score = silhouette_score(sample, kmeans.labels_)
        silhouette_scores.append(score)




    plt.figure(figsize=(8, 6))
    plt.plot(range(2, 11), silhouette_scores, marker='o')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Score de silhouette KMeans')
    plt.title('Score de silhouette pour différents nombres de clusters')
    plt.show()

    return silhouette_scores


def scaler_code(df) :
    m = (df['NB_ODD_15_TC'].mean())
    print(m)

    s = df['NB_ODD_15_TC'].std()*np.sqrt(4/5)
    print((df['NB_ODD_15_TC']-m)/s)

    return True

def scale_data(data): 

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    return(data)

def inverse_scale_data(data):

    scaler = StandardScaler()

    scaler.fit(data)
    data = scaler.transform(data)

    data = scaler.inverse_transform(data)
    data = np.round(data).astype(int)

    return data

def scale_data_fit(data): 

    scaler = StandardScaler()
    scaler.fit(data)

    data = scaler.transform(data)

    return data
    


def init_scaler(data):

    scaler = StandardScaler()
    scaler.fit(data)

    return True 

def silouhette_scores(data,range_n_clusters):

    d = {}

    for n_clusters in range_n_clusters:

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(data)

        silhouette_avg = silhouette_score(data, cluster_labels)
        print(
            "For n_clusters =", n_clusters,
            "The average silhouette_score is :", silhouette_avg,
        )

        d[n_clusters] = silhouette_avg

    return d 


def visualize_silhouette_datas(data,range_n_clusters):
    
    for n_clusters in range_n_clusters:

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(data)

        silhouette_avg = silhouette_score(data, cluster_labels)
        print(
            "For n_clusters =", n_clusters,
            "The average silhouette_score is :", silhouette_avg,
        )

        fig, ax1 = plt.subplots(1, 1)  # Modification ici : un seul subplot au lieu de deux
        fig.set_size_inches(9, 7)

    # Le reste du code reste inchangé


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

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.show()

    return True 

def visualize_silhouette_datas_pdf(data, range_n_clusters, output_filename):


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