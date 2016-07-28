import csv
import argparse
import pandas
import numpy

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

# order by correlation
corr = wc_corr.unstack().order(kind="quicksort").to_frame()
print corr[corr.index.get_level_values(0) != corr.index.get_level_values(1)][corr[0]>.9]
