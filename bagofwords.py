# run in directory with wordcount files ending in .w.txt (created by bagofwords.sh)

import pandas
import os
import pprint
import matplotlib.pylab as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans

# Convert wordcount files to list of dictionaries for each page
wordcounts_f = [f for f in os.listdir('.') if f.endswith('.w.txt')]
wordcounts = []
for fl in wordcounts_f:
	wc_dict = dict()
	for line in open(fl, 'r'):
		count = line.strip().partition(' ')
		if count[2]: # drop empty lines
			wc_dict[count[2]] = int(count[0])
	wordcounts.append(wc_dict)

# Use DictVectorizer to convert the dictionaries to a sparse array like sklearn wants
word_features = DictVectorizer().fit_transform(wordcounts).toarray()
#print word_features.get_feature_names()

# k-means clustering
clusters = KMeans(n_clusters=26).fit_predict(word_features)
print clusters

# Plot clusters
plt.scatter(range(1,len(clusters)+1), clusters)
plt.show()
