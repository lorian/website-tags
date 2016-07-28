import csv
import argparse
import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn as sns

bad_words = ['code', 'japan', 'help', 'contacted', 'labels', 'touch', 'tcg', 'slot', 'fibre', 'ads', 'giftcard', 'ships', 'here', 'bfr', 'add', 'black', 'th', 'careers', 'return', 'get', 'dell', 'touchpad', 'datasheets', 'rewardterms', 'using', 'itanium', 'now', 'emails', 'determines', 'level', 'taxes', 'standards', 'separately', 'fgeneric', 'mhz', 'payments', 'dpa', 'small', 'page', 'www', 'generation', 'spring', 'configured', 'spanish', 'back', 'tvs', 'pvc', 'year', 'logo', 'btn', 'odd', 'click', 'rw', 'dense', 'ddr', 'please', 'legal', 'label', 'safety', 'finger', 'reader', 'got', 'net', 'integrated', 'participation', 'full', 'terms', 'business', 'equotes', 'cart', 'bt', 'computrace', 'lcd', 'free', 'trademarks', 'promoterms', 'solid', 'optional', 'certified', 'onsite', 'hours', 'unresolved', 'license', 'spill', 'standby', 'compliance', 'shipping', 'ltd', 'vault', 'processor', 'pays', 'primary', 'webbank', 'apply', 'total', 'forums', 'minitower', 'sheet', 'billing', 'capable', 'monthly', 'dfs', 'width', 'refurbished', 'memory', 'pf', 'assessment', 'box', 'life', 'buy', 'form', 'kb', 'hr', 'wi', 'masthead', 'optical', 'hd', 'applied', 'financing', 'comparecompare', 'anything', 'mm', 'double', 'mo', 'widescreen', 'enabled', 'sdxc', 'hdd', 'arrive', 'purchases', 'lbs', 'required', 'investors', 'lock', 'premier', 'egift', 'close', 'wwan', 'mouse', 'uncheck', 'different', 'newsroom', 'drivers', 'nd', 'damage', 'configurable', 'member', 'internal', 'pad', 'sdhc', 'contactless', 'sim', 'phi', 'arrives', 'diagnosis', 'customize', 'price', 'loyalty', 'weee', 'sff', 'fi', 'law', 'approved', 'webcam', 'kg', 'charges', 'tpm', 'register', 'drive', 'order', 'savings', 'responsibility', 'workspace', 'typically', 'mic']

parser = argparse.ArgumentParser(description='Imports wordcounts from csv and determines word correlations')
parser.add_argument('wordcount_file', help='csv containing table of wordcounts per page')
args = parser.parse_args()

# import wordcounts
with open(args.wordcount_file,'r') as wordcount_file:
	wordcount_data = [r for r in csv.reader(wordcount_file, delimiter=' ')]
	wc = pandas.DataFrame(wordcount_data)

# label dataframe
wc.columns = wc.iloc[0]
#wc.drop(bad_words, axis=1, inplace=True) # drop previously-created stop list
wc.drop('',1, inplace=True)
wc.drop(wc.index[:1], inplace=True)

# make correlation matrix between columns (words)
wc_nums = wc.drop('URL',1).astype(float)
wc_corr = wc_nums.corr()

# plot correlations of top 50 words
wc_top = wc_nums.ix[:,0:50]
correlations = wc_top.corr()

'''
fig = plt.figure()
ax = fig.add_subplot(111)
sns.heatmap(correlations, vmin=-1, vmax=1, square=True)
ax.xaxis.tick_top()
ax.set_xticklabels(wc_top.columns, rotation=-90)
ax.set_yticklabels(reversed(wc_top.columns), rotation=0)
ax.set_xlabel('')
ax.set_ylabel('')
plt.show()
'''

# order by correlation; make list of words to drop
keep = set()
drop = set()
corr = wc_corr.unstack().order(kind="quicksort").to_frame()
corr = corr[corr.index.get_level_values(0) != corr.index.get_level_values(1)] # remove diagonal line

output = corr[corr[0]>.85]
pairs = zip(output.index.get_level_values(0),output.index.get_level_values(1))
for pair in pairs:
	if pair[0] not in keep:
		drop.add(pair[0])
		keep.add(pair[1])
	elif pair[1] not in keep:
		drop.add(pair[1])
		keep.add(pair[0])

print list(drop)

# find words that have least difference between min and max variance
min_words = wc_corr.min(axis=1)
numpy.fill_diagonal(wc_corr.values, -2) # remove diagonal
max_words = wc_corr.max(axis=1)
minmax_words = (max_words-min_words).order(kind="quicksort")[0:100].index

# find words that are equally correlated with everything
var_words = wc_corr.var(axis=1).order(kind="quicksort")[0:100].index

# pick one to plot/print
even_corrs = wc_corr[var_words]
even_corrs = wc_corr[minmax_words]
overlap_words = list(set(var_words).intersection(minmax_words))
even_corrs = wc_corr[overlap_words]

print overlap_words

fig = plt.figure()
ax = fig.add_subplot(111)
sns.heatmap(even_corrs, vmin=-1, vmax=1, yticklabels=False)
ax.xaxis.tick_top()
ax.set_xticklabels(even_corrs.columns, rotation=-90)
ax.set_xlabel('')
ax.set_ylabel('')
#plt.show()

