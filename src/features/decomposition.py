import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import src.data.datasets as ds
import src.data.train_test_split as split
from sklearn.decomposition import PCA
from pathlib import Path



def pca(name, X_train, X_test):
	"""

	"""

	X_train = X_train.copy()
	X_test = X_test.copy()
	num_numerical = ds.get_number_numerical(name)
	X_train_s = split.standardize(name, X_train)
	X_test_s = split.standardize(name, X_test)
	X_train_s_numerical = X_train_s.iloc[:,0:num_numerical]
	X_train_s_categorical = X_train_s.iloc[:,num_numerical:]
	X_test_s_numerical = X_test_s.iloc[:,0:num_numerical]
	X_test_s_categorical = X_test_s.iloc[:,num_numerical:]
	estimator = PCA(0.95)
	X_train_s_numerical_reduced = pd.DataFrame(estimator.fit_transform(X_train_s_numerical), 
	                                           index = X_train_s_categorical.index)
	X_test_s_numerical_reduced = pd.DataFrame(estimator.transform(X_test_s_numerical), 
	                                          index = X_test_s_categorical.index)
	X_train_s = pd.concat([X_train_s_numerical_reduced, X_train_s_categorical], axis = 1)
	X_test_s = pd.concat([X_test_s_numerical_reduced, X_test_s_categorical], axis = 1)
	return X_train_s, X_test_s


def pca_component_analysis(name, X_train, save = False):
	"""

	"""

	X_train = X_train.copy()
	num_numerical = ds.get_number_numerical(name)
	X_train_s = split.standardize(name, X_train).iloc[:,0:num_numerical]

	pca = PCA().fit(X_train_s)

	fig, ax = plt.subplots(figsize=(20, 8))
	ax.plot(np.cumsum(pca.explained_variance_ratio_), marker = 'o')
	ax.tick_params(labelsize=15)
	ax.set_xlabel('Number of Components', fontsize=20)
	ax.set_ylabel('Cumulative Explained Variance', fontsize=20);

	if save:
		to_save = Path().resolve().joinpath('data', 'visualizations', '{}_pca_component_analysis.png'.format(name))
		fig.savefig(to_save)  
