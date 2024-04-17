from clustering_Kmeans import * 

data_path = "/Users/maximecoppa/Desktop/Statapp_Data/Datas/"  # Path to data directory


# Test Kmeans
trace_silhouette_scores(visualize_silhouette_datas_all(data_path + "data_clustering.csv",
                                                       ['ODD 15 jours TC_MEAN_TIME_DIFF'],
                                                       [2, 3, 4, 5],
                                                       data_path + "Clustering_results/"), [2, 3, 4,5])
