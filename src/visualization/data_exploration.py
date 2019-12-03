import src.data.datasets as ds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path


def create_all_plots(name, train):
	'''Plots all the different plotting functions in one call.

	'''
	train = train.copy()
	create_attendance_histogram(name, train)
	create_daily_histogram(name, train)
	create_daily_barchart(name, train)
	create_monthly_histogram(name, train)
	create_monthly_barchart(name, train)
	create_yearly_histogram(name, train)
	create_yearly_barchart(name, train)
	create_playoffs_histograms(name, train)
	create_win_percent_histograms(name, train)
	create_last_five_record_histograms(name, train)
	create_last_five_record_barchart(name, train)
	create_heatmap(name, train)

def create_attendance_histogram(name, train, save = False):
	'''Creates and/or saves histogram of attendance for a given train as a subplot.

	'''
	sns.set_style("whitegrid")
	sns.set_palette(sns.color_palette('bright',12))

	bins = np.arange(4000, 26001, 500)
	fig, ax = plt.subplots(figsize=(20, 8))
	sns.distplot(train['Attendance'], ax=ax, kde=False, norm_hist=True, bins = bins)
	ax.set_xlabel('Attendance', fontsize = 20)
	ax.set_ylabel('Percent per Attendance', fontsize = 20) 
	ax.tick_params(labelsize=15)
	ax.set_title(label = "NBA Attendance for {}".format(name), fontsize = 25)
	plt.show()

	if save:
		to_save = Path().resolve().joinpath('data', 'visualizations', 'overall_attendance_hist_{}.png'.format(name))
		fig.savefig(to_save)

def create_daily_histogram(name, train, save = False):
	'''Creates and/or saves overlaid histograms of daily attendance as a subplot.

	'''

	sns.set_style("whitegrid")
	sns.set_palette(sns.color_palette('bright',12))

	bins = np.arange(4000, 26001, 500)
	fig, ax = plt.subplots(figsize=(20, 8))
	days = train['Day of Week'].unique()
	for day in days:
		sns.distplot(train.loc[train['Day of Week'] == day]['Attendance'],
							ax = ax, kde= False, norm_hist= True, bins = bins)
	ax.set_title(label = "NBA Attendance per Day for {}".format(name), fontsize = 25)
	ax.set_xlabel('Attendance', fontsize = 20)
	ax.set_ylabel('Percent per Attendance', fontsize = 20) 
	ax.tick_params(labelsize=15)
	ax.legend(days,loc="upper right", fontsize=20)
	plt.show()

	if save:
		fig.savefig('daily_attendance_hist_{}.png'.format(name))

def create_daily_barchart(name, train, save = False): 
	'''Creates and/or saves horizontal bar chart of mean attendance per day as a subplot.

	'''

	fig, ax = plt.subplots(figsize=(20, 8))
	grouped = train[['Day of Week', 'Attendance']].groupby('Day of Week').mean()
	order = ['Monday', "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
	sns.barplot(x='Attendance', y=grouped.index, data=grouped, order=order, ci=None, orient='h', 
	             saturation=1, ax=ax, palette = sns.color_palette("cubehelix", 7))
	ax.set_xlim(16500,19000)
	ax.set_xticks(range(16500,19001,500))
	ax.set_xlabel('Average Attendance', fontsize = 20)
	ax.set_ylabel('Day of the Week', fontsize = 20) 
	ax.tick_params(labelsize=15)
	ax.set_title(label = "Average NBA Attendance per Day for {}".format(name), fontsize = 25)
	plt.show()

	if save:
		to_save = Path().resolve().joinpath('data', 'visualizations', 'daily_attendance_barchart_{}.png'.format(name))
		fig.savefig(to_save)

def create_monthly_histogram(name, train, save = False):
	'''Creates and/or saves overlaid histograms of monthly attendance as a subplot

	'''
	sns.set_style("whitegrid")
	sns.set_palette(sns.color_palette('bright',12))
	
	bins = np.arange(4000, 26001, 500)
	fig, ax = plt.subplots(figsize=(20, 8))
	months = train['Month'].unique()
	for month in months:
		sns.distplot(train.loc[train['Month'] == month]['Attendance'],
							ax = ax, kde= False, norm_hist= True, bins = bins)
	ax.set_title(label = "NBA Attendance per Month for {}".format(name), fontsize = 25)
	ax.set_xlabel('Attendance', fontsize = 20)
	ax.set_ylabel('Percent per Attendance', fontsize = 20) 
	ax.tick_params(labelsize=15)
	ax.legend(months,loc="upper right", fontsize=20)
	plt.show()

	if save:
		to_save = Path().resolve().joinpath('data', 'visualizations', 'monthly_attendance_hist_{}.png'.format(name))
		fig.savefig(to_save)

def create_monthly_barchart(name, train, save = False): 
	'''Creates and/or saves horizontal bar chart of mean attendance per day as a subplot.

	'''

	fig, ax = plt.subplots(figsize=(20, 8))
	grouped = train[['Month', 'Attendance']].groupby('Month').mean()
	order = ['October', 'November', 'December', 'January','February', 'March', 'April', 'May', 'June']
	sns.barplot(x='Attendance', y=grouped.index, data=grouped, order=order, ci=None, orient='h', 
	             saturation=1, ax=ax, palette = sns.color_palette("cubehelix", 9))
	ax.set_xlim(16500,19500)
	ax.set_xticks(range(16500,19501,500))
	ax.set_xlabel('Average Attendance', fontsize = 20)
	ax.set_ylabel('Month', fontsize = 20) 
	ax.tick_params(labelsize=15)
	ax.set_title(label = "Average NBA Attendance per Month for {}".format(name), fontsize = 25)
	plt.show()

	if save:
		to_save = Path().resolve().joinpath('data', 'visualizations', 'monthly_attendance_barchart_{}.png'.format(name))
		fig.savefig(to_save)


def create_yearly_histogram(name, train, save = False, years = np.arange(2014,2020)):
	'''Creates and/or saves overlaid histograms of yearly attendance of the last five years as a subplot.

	'''
	sns.set_style("whitegrid")
	sns.set_palette(sns.color_palette('bright',12))
	
	bins = np.arange(4000, 26001, 500)
	fig, ax = plt.subplots(figsize=(20, 8))
	for year in years:
		sns.distplot(train.loc[train.index.year == year]['Attendance'],
							ax = ax, kde= False, norm_hist= True, bins = bins)
	ax.set_title(label = "NBA Attendance per Year for {}".format(name), fontsize = 25)
	ax.set_xlabel('Attendance', fontsize = 20)
	ax.set_ylabel('Percent per Attendance', fontsize = 20) 
	ax.tick_params(labelsize=15)
	ax.legend(years,loc="upper right", fontsize=20)
	plt.show()

	if save:
		to_save = Path().resolve().joinpath('data', 'visualizations', 'yearly_attendance_hist_{}.png'.format(name))
		fig.savefig(to_save)

def create_yearly_barchart(name, train, save = False): 
	'''Creates and/or saves horizontal bar chart of mean attendance per day as a subplot.

	'''

	fig, ax = plt.subplots(figsize=(20, 8))
	train['Year'] = train.index.year
	grouped = train[['Year','Attendance']].groupby('Year').mean()
	order = np.arange(2004,2017)
	sns.barplot(x='Attendance', y=grouped.index, data=grouped, order=order, ci=None, orient='h', 
	             saturation=1, ax=ax, palette = sns.color_palette("cubehelix", 13))
	ax.set_xlim(16500,18500)
	ax.set_xticks(range(16500,18501,500))
	ax.set_xlabel('Average Attendance', fontsize = 20)
	ax.set_ylabel('Year', fontsize = 20) 
	ax.tick_params(labelsize=15)
	ax.set_title(label = "Average NBA Attendance per Year for {}".format(name), fontsize = 25)
	plt.show()

	if save:
		to_save = Path().resolve().joinpath('data', 'visualizations', 'yearly_attendance_barchart_{}.png'.format(name))
		fig.savefig(to_save)

def create_playoffs_histograms(name, train, save = False):
	'''Creates and/or saves overlaid histograms of attendance for playoff versus regular games as a subplot.

	'''
	sns.set_style("whitegrid")
	sns.set_palette(sns.color_palette('bright',12))
	
	bins = np.arange(4000, 26001, 500)
	fig, ax = plt.subplots(figsize=(20, 8))
	for j in [0,1]:
		sns.distplot(train.loc[train['Playoffs?'] == j]['Attendance'],
							ax = ax, kde= False, norm_hist= True, bins = bins)
	ax.set_title(label = "NBA Attendance for Regular and Playoff Games for {}".format(name), fontsize = 25)
	ax.set_xlabel('Attendance', fontsize = 20)
	ax.set_ylabel('Percent per Attendance', fontsize = 20) 
	ax.tick_params(labelsize=15)
	ax.legend([0,1],loc="upper right", fontsize=20)
	plt.show()

	if save:
		to_save = Path().resolve().joinpath('data', 'visualizations', 'playoffs_attendance_hist_{}.png'.format(name))
		fig.savefig(to_save)

def create_win_percent_histograms(name, train, save = False):
	'''Creates and/or saves overlaid histograms of attendance for different winning percentages 
	as a subplot.

	'''
	sns.set_style("whitegrid")
	sns.set_palette(sns.color_palette('bright',12))

	win_percent = np.arange(0,1,0.10)
	bins = np.arange(4000, 26001, 500)
	fig, ax = plt.subplots(figsize=(20, 8))
	for i in win_percent:
		sns.distplot(train.loc[(train['Curr Win %'] >= i) & (train['Curr Win %'] < i+0.1)]['Attendance'],
                        ax=ax, kde=False, norm_hist=True, bins = bins)
	ax.set_title(label = "NBA Attendance for Different Win Percentages for {}".format(name), fontsize = 25)
	ax.set_xlabel('Attendance', fontsize = 20)
	ax.set_ylabel('Percent per Attendance', fontsize = 20) 
	ax.tick_params(labelsize=15)
	ax.legend(np.round(win_percent,1),loc="upper right", fontsize=20)
	plt.show()

	if save:
		to_save = Path().resolve().joinpath('data', 'visualizations', 'win_percentage_attendance_hist_{}.png'.format(name))
		fig.savefig(to_save)

def create_last_five_record_histograms(name, train, save = False):
	'''Creates and/or saves overlaid histograms of attendance for different winning percentages 
	as a subplot.

	'''
	sns.set_style("whitegrid")
	sns.set_palette(sns.color_palette('bright',12))
	
	last_five = np.arange(0,6)
	bins = np.arange(4000, 26001, 500)
	fig, ax = plt.subplots(figsize=(20, 8))
	for i in last_five:
		sns.distplot(train.loc[train['Last Five'] == i]['Attendance'],
                        ax=ax, kde=False, norm_hist=True, bins = bins)
	ax.set_title(label = "NBA Attendance by Last Five Record for {}".format(name), fontsize = 25)
	ax.set_xlabel('Attendance', fontsize = 20)
	ax.set_ylabel('Percent per Attendance', fontsize = 20) 
	ax.tick_params(labelsize=15)
	ax.legend(last_five,loc="upper right", fontsize=20)
	plt.show()

	if save:
		to_save = Path().resolve().joinpath('data', 'visualizations', 'last_five_record_attendance_hist.{}.png'.format(name))
		fig.savefig(to_save)

def create_last_five_record_barchart(name, train, save = False): 
	'''Creates and/or saves horizontal bar chart of mean attendance per day as a subplot.

	'''

	fig, ax = plt.subplots(figsize=(20, 8))
	train['Year'] = train.index.year
	grouped = train[['Last Five','Attendance']].groupby('Last Five').mean()
	order = np.arange(0,6)
	sns.barplot(x='Attendance', y=grouped.index, data=grouped, order=order, ci=None, orient='h', 
	             saturation=1, ax=ax, palette = sns.color_palette("cubehelix", 5))
	ax.set_xlim(16500,18500)
	ax.set_xticks(range(16500,18501,500))
	ax.set_xlabel('Average Attendance', fontsize = 20)
	ax.set_ylabel('Record Over Last Five Games', fontsize = 20) 
	ax.tick_params(labelsize=15)
	ax.set_title(label = "Average NBA Attendance per Last Five Record for {}".format(name), fontsize = 25)
	plt.show()

	if save:
		to_save = Path().resolve().joinpath('data', 'visualizations', 'last_five_attendance_barchart_{}.png'.format(name))
		fig.savefig(to_save)

def create_heatmap(name, train, save = False):
	'''Creates and/or saves a heatmap of the correlation between the numerical variables of a train.

	'''
	train = train.copy()
	num_numerical = ds.get_number_numerical(name)
	train_num = train.iloc[:,0:num_numerical]
	fig, ax = plt.subplots(figsize=(15, 15))
	heat = sns.heatmap(train_num.corr(),annot = True, ax = ax, fmt = '.2f', 
	                   cbar = True, square = True, xticklabels= True, yticklabels = True,
	                  annot_kws={'size':16}, cmap = 'coolwarm', center= 0, vmin=-1, vmax=1,
	                  cbar_kws={"shrink": .82})
	ax.set_title('Heatmap of Numerical Variable Correlation for {}'.format(name), size=25) 
	plt.yticks(rotation=0,size = 15) 
	plt.xticks(rotation=30, size = 15)
	ax.collections[0].colorbar.ax.tick_params(labelsize=15)

	# Make annotations larger if abs(correlation) above 0.2
	num_corrs = len(np.unique(train_num.corr().values.flatten()))
	bigs = []
	for i in np.arange(2,num_corrs+1):
	    val = round(np.sort(np.abs(np.unique(train_num.corr().values.flatten())))[-i],2)
	    if val > 0.2:
	        bigs = np.append(bigs, val)
	for text in heat.texts:
	    num =  pd.to_numeric(text.get_text())
	    i = np.where(bigs == abs(num))[0]
	    if i.size > 0:
	        text.set_color('white')
	        text.set_size(40-(i[0]*3))
	plt.show()  

	if save:
		to_save = Path().resolve().joinpath('data', 'visualizations', '{}_heatmap.png'.format(name))
		fig.savefig(to_save)   




