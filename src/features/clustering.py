import matplotlib.pyplot as plt
import pandas as pd
import src.data.datasets as ds
import src.data.train_test_split as split
from pathlib import Path
from sklearn.cluster import KMeans


def kmeans(name, n_clusters, X_train):
	'''

	'''

	X_train = X_train.copy()
	num_numerical = ds.get_number_numerical(name)
	X_train_s_numerical = split.standardize(name, X_train).iloc[:,0:num_numerical]
	return KMeans(n_clusters= n_clusters, random_state=18, n_jobs = 1).fit(X_train_s_numerical)

def elbow_method_kmeans(name, X_train, max_clusters = 40, save = False):
	'''Creates elbow method plot for varying number of clusters.

	Y-axis is the sum of squared distances of samples to their closest cluster center.
	'''

	X_train = X_train.copy()
	num_numerical = ds.get_number_numerical(name)
	X_train_s_numerical = split.standardize(name, X_train).iloc[:,0:num_numerical]
	distortions = []
	cluster_range = range(2,max_clusters)
	for clusters in cluster_range:
		kmean = kmeans(name, clusters, X_train_s_numerical)
		distortions.append(kmean.inertia_)

	fig, ax = plt.subplots(figsize=(15, 15))
	ax.plot(cluster_range, distortions, marker = 'o')
	for i, ann in enumerate(cluster_range):
		ax.annotate(ann, (cluster_range[i], distortions[i]))
	ax.set_title('Elbow Method for KMeans for {}'.format(name), size=25)
	ax.set_xlabel('Number of Clusters', fontsize = 20)
	ax.set_ylabel('Sum of Squared Distances', fontsize = 20) 
	plt.show()

	if save:
		to_save = Path().resolve().joinpath('data', 'visualizations', '{}_elbow.png'.format(name))
		fig.savefig(to_save) 
