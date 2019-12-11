import pandas as pd
import src.data.datasets as ds
import src.data.train_test_split as split
import src.features.clustering as clustering
import src.features.decomposition as decomposition
import src.features.statistical_tests as st
import src.initialize_jupyter
import src.models.ensemble_models as ensembles
import src.models.linear_models as linear_models
import src.models.metrics as metrics
import src.models.neural_networks as nn
import src.models.other_models as other_models
import src.visualization.data_exploration as de
from IPython.display import display, Image, Markdown

def main_datasets():
	'''

	'''

	pd.set_option('display.max_rows', 2)
	number_numerical = ds.get_number_numerical()
	dsets = ds.load_datasets(names = ['dataset_1', 'dataset_2', 'dataset_3'])
	display(Markdown('### `Dataset 1:` {} features: {} numerical, {} categorical, 1 response'.format(\
	                                                                                     len(dsets['dataset_1'].columns)-1, 
	                                                                                     number_numerical['dataset_1'], 
	                                                                                     len(dsets['dataset_1'].columns)\
	                                                                                     -1-number_numerical['dataset_1'])))
	display(dsets['dataset_1'])
	display(Markdown('---'))
	display(Markdown('### `Dataset 2:` {} features: {} numerical, {} categorical, 1 response'.format(\
	                                                                                     len(dsets['dataset_2'].columns)-1, 
	                                                                                     number_numerical['dataset_2'], 
	                                                                                     len(dsets['dataset_2'].columns)\
	                                                                                     -1-number_numerical['dataset_2'])))
	display(dsets['dataset_2'])
	display(Markdown('---'))
	display(Markdown('### `Dataset 3:` {} features: {} numerical, {} categorical, 1 response'.format(\
	                                                                                     len(dsets['dataset_3'].columns)-1, 
	                                                                                     number_numerical['dataset_3'], 
	                                                                                     len(dsets['dataset_3'].columns)\
	                                                                                     -1-number_numerical['dataset_3'])))
	display(dsets['dataset_3'])
	display(Markdown('---'))

def main_datasets_split():
	'''

	'''

	number_numerical = ds.get_number_numerical()
	pd.set_option('display.max_rows', 2)
	dsets = split.split_subsets(['dataset_1', 'dataset_2', 'dataset_3'])
	display(Markdown('### `Dataset 1 Training Set:` {} features: {} numerical, {} categorical, 1 response'.format(\
	                                                                                     len(dsets['dataset_1'][0].columns)-1, 
	                                                                                     number_numerical['dataset_1'], 
	                                                                                     len(dsets['dataset_1'][0].columns)\
	                                                                                     -1-number_numerical['dataset_1'])))
	display(dsets['dataset_1'][0])
	display(Markdown('---'))
	display(Markdown('### `Dataset 2 Training Set:` {} features: {} numerical, {} categorical, 1 response'.format(\
	                                                                                     len(dsets['dataset_2'][0].columns)-1, 
	                                                                                     number_numerical['dataset_2'], 
	                                                                                     len(dsets['dataset_2'][0].columns)\
	                                                                                     -1-number_numerical['dataset_2'])))
	display(dsets['dataset_2'][0])
	display(Markdown('---'))
	display(Markdown('### `Dataset 3 Training Set:` {} features: {} numerical, {} categorical, 1 response'.format(\
	                                                                                     len(dsets['dataset_3'][0].columns)-1, 
	                                                                                     number_numerical['dataset_3'], 
	                                                                                     len(dsets['dataset_3'][0].columns)\
	                                                                                     -1-number_numerical['dataset_3'])))
	display(dsets['dataset_3'][0])
	display(Markdown('---'))

def plots():
	'''Plots all the different plotting functions in one call.

	'''
	display(Markdown('### `Dataset 1:`'))
	display(Image("data/visualizations/all_plots_dataset_1.png"))
	display(Markdown('---'))
	display(Markdown('### `Dataset 2:`'))
	display(Image("data/visualizations/all_plots_dataset_2.png"))
	display(Markdown('---'))
	display(Markdown('### `Dataset 3:`'))
	display(Image("data/visualizations/all_plots_dataset_3.png"))
	display(Markdown('---'))


