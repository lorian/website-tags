# run in directory with wordcount files ending in .w.txt (created by bagofwords.sh)

import pandas
import os
import pprint
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer

wordcounts_f = [f for f in os.listdir('.') if f.endswith('.w.txt')]
wordcounts = []
for fl in wordcounts_f:
	wc_dict = dict()
	wc_dict['File'] = fl
	for line in open(fl, 'r'):
		count = line.strip().partition(' ')
		wc_dict[count[2]] = int(count[0])
	wordcounts.append(wc_dict)
		
#pprint.pprint(wordcounts)

vec = DictVectorizer()
vec.fit_transform(wordcounts).toarray()
print vec.get_feature_names()

