import matplotlib.pyplot as plt


def pca(name, X_train, y_trainsave = False):
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
