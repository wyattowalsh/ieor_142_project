import pandas as pd
import src.data.train_test_split as split
import src.features.decomposition as decomposition
import src.models.metrics as metrics
from functools import partial
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor


def collect_statistics(name, X_train, X_test, y_train, y_test):

	X_train = X_train.copy()
	X_test = X_test.copy()
	y_train = y_train.copy()
	y_test = y_test.copy()
	X_train_s = split.standardize(name,X_train)
	X_test_s = split.standardize(name, X_test)
	X_train_pca, X_test_pca = decomposition.pca(name,X_train, X_test)

	variation_strings = ['', ' with Standardized Features', ' with PCA']
	variation_training = [X_train, X_train_s, X_train_pca]
	variation_test = [X_test, X_test_s, X_test_pca]
	models = [simple_network]
	model_names = ["Simple Network"]
	regression_statistics = pd.DataFrame()
	for i,model in enumerate(models):
		models_preds = list(map(partial(model, y_train = y_train), variation_training, variation_test))
		model_name = model_names[i]
		for j, variation in enumerate(variation_strings):
			model_metrics = metrics.apply_metrics('{} {}{}'.format(name, model_name, variation), 
			                                      y_test, models_preds[j].ravel())
			regression_statistics = pd.concat([regression_statistics, model_metrics], axis = 0)

	return regression_statistics


def simple_network(X_train, X_test, y_train):
	"""

	"""

	net = Sequential()
	net.add(Dense(15, activation='relu', input_shape=(X_train.shape[1],)))
	net.add(Dense(1, activation='linear'))
	optimizer = Adam()
	net.compile(optimizer=optimizer, loss='mean_squared_error')
	net.fit(X_train, y_train,  epochs=50, batch_size=32, verbose = 0, shuffle = False)
	predictions = net.predict(X_test)
	return predictions












# def create_net(neurons=10, neurons_hl=10, layers=0, learning_rate=0.001, activations_1='relu', activations_2='relu'):
#     net = Sequential()
#     net.add(Dense(neurons, activation=activations_1,
#                   input_shape=(X_train.shape[1],)))
#     for i in np.arange(layers):
#         net.add(Dense(neurons_hl, activation=activations_2))
#     net.add(Dense(1, activation='linear'))
#     # learning_rate=learn_rate
#     optimizer = Adam(learning_rate=learning_rate)
#     net.compile(optimizer=optimizer, loss='mean_squared_error')
#     return net


# neurons = np.arange(10, 101, 5)
# neurons_hl = np.arange(5, 101, 5)
# layers = [0, 1]
# batch_size = [64]
# epochs = [100]
# acts = ['relu']
# learn_rate = [0.001, 0.01, 0.05, .1, 1]
# param_grid = dict(neurons=neurons, neurons_hl=neurons_hl, layers=layers,
#                   epochs=epochs, batch_size=batch_size, activations_1=acts,
#                   activations_2=acts, learning_rate=learn_rate)
# net = KerasRegressor(build_fn=create_net, verbose=0)
# grid = RandomizedSearchCV(estimator=net, param_distributions=param_grid, n_jobs=-1, cv=5, refit='neg_mean_absolute_error',
#                           n_iter=200, iid=False, scoring=['neg_mean_absolute_error', 'r2'])
# grid_result = grid.fit(X_train, y_train, shuffle=False)