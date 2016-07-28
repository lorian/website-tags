import csv
import argparse
import pandas
import numpy
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Imports wordcounts from csv and determines word correlations')
parser.add_argument('wordcount_file', help='csv containing table of wordcounts per page')
args = parser.parse_args()

# import wordcounts
with open(args.wordcount_file,'r') as wordcount_file:
	wordcount_data = [r for r in csv.reader(wordcount_file, delimiter=' ')]
	wc = pandas.DataFrame(wordcount_data)

# label dataframe
wc.columns = wc.iloc[0]
wc.drop('',1, inplace=True)
wc.drop(wc.index[:1], inplace=True)

# make correlation matrix between columns (words)
wc_nums = wc.drop('URL',1).astype(float)
wc_corr = wc_nums.corr()

# plot correlations of top 50 words
wc_top = wc_nums.ix[:,0:50]
correlations = wc_top.corr()

fig = plt.figure()
ax = fig.add_subplot(111)
fig.colorbar(ax.matshow(correlations, vmin=-1, vmax=1))
ax.set_xticks(numpy.arange(correlations.shape[1]))
ax.set_yticks(numpy.arange(correlations.shape[0]))
ax.set_xticklabels(wc_top.columns, rotation=90)
ax.set_yticklabels(wc_top.columns)
plt.show()

# order by correlation
corr = wc_corr.unstack().order(kind="quicksort").to_frame()
output = corr[corr.index.get_level_values(0) != corr.index.get_level_values(1)][corr[0]>.9]

