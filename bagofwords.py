# run in directory with wordcount files ending in .w.txt (created by bagofwords.sh)

import pandas
import os
import pprint
import argparse
import matplotlib.pylab as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans
import webbrowser
import subprocess
import collections
import string
import re
import itertools

# Script takes number of pages and number of clusters as arguments, has optional flags for ways to look at clusters
parser = argparse.ArgumentParser(description='Imports wordcounts from webpages and clusters them')
parser.add_argument('num_pages', help='How many webpages to analyze and cluster (max of 500)')
parser.add_argument('num_clusters', help='How many clusters to group pages into')
parser.add_argument('--output', help='Display clusters as [plot] or see webpages in [browser]')
parser.add_argument('--meta', help="Which meta tag to judge clusters by. Interesting options include og:title, Description, Keywords, og:type, CategoryPath, PageType, SalesType.")
parser.add_argument('--file-output', action='store_true', help="Flag will output cluster details to a file. Requires --meta to be set.")
args = parser.parse_args()

# Convert wordcount files to list of dictionaries for each page
wordcounts_f = [f for f in os.listdir('.') if f.endswith('.w.txt')]
wordcounts_f.sort(reverse=True) # makes sure labeled pages are present by default
wordcounts_f = wordcounts_f[0:int(args.num_pages)]
wordcounts = []
for fl in wordcounts_f:
	wc_dict = dict()
	for line in open(fl, 'r'):
		count = line.strip().partition(' ')
		if count[2]: # drop empty lines
			wc_dict[count[2]] = int(count[0])
	wordcounts.append(wc_dict)

# Use DictVectorizer to convert the dictionaries to a sparse array for sklearn
word_features = DictVectorizer().fit_transform(wordcounts).toarray()

# K-means clustering
clusters = KMeans(n_clusters=int(args.num_clusters)).fit_predict(word_features)
clustered_pages = zip(clusters,wordcounts_f)
clustered_pages.sort(key=lambda x: x[0])

def display(text):
	# print to screen or prints to file, depending on options set
	if args.file_output:
		f.write(text +'\n')
	else:
		print text

# Display cluster information
cluster_counts = collections.Counter(clusters)
if args.meta:
	if args.file_output:
		f = open('../clusters_by_{}.txt'.format(args.meta.translate(string.maketrans("",""), string.punctuation)),'w')
	
	display("\tCluster overview:")
	for k,v in cluster_counts.most_common():
		display("{} {}".format(k,v))
		
	# get titles from all pages per cluster
	for cl in cluster_counts.keys():
		cluster_metadata = []
		display("\n\tCluster {} ({} pages):".format(cl,cluster_counts[cl]))

		for c,page in clustered_pages:
			if cl == int(c):
				webpage = page.partition('.')[0]+'.html'
				# List page content from metadata
				title = subprocess.Popen('grep "{}" {}'.format(args.meta, webpage), shell=True, stdout=subprocess.PIPE)
				# gets only content, removes "Dell" tag on the end of some, then removes quotes and spaces at ends
				cluster_metadata.append(title.communicate()[0].partition('content=')[2].partition('/>')[0].partition('| Dell')[0].strip().strip('"').strip().lower())
		
		# check how many pages had metadata of this type
		has_content = collections.Counter(cluster_metadata)
		if has_content['']:
			display("\tPages without this metadata: {}".format(has_content['']))
		
		# count words; drop words with a frequency of 1 or symbols/numbers
		if args.meta == 'CategoryPath': # special format
			cluster_metadata = [item for sublist in [w.split('/') for w in cluster_metadata] for item in sublist]
		all_words = " ".join(cluster_metadata).translate(string.maketrans("",""), string.punctuation).split(' ')
		bag_words = collections.Counter({k: c for k, c in collections.Counter(all_words).items() if k.isalpha()})

		for k,v in bag_words.most_common():
			if cluster_counts[cl] >1 and v <2 and args.meta in ['Title', 'Description', 'Keywords', 'CategoryPath']: # filter out words that appear only once
				pass;
			else:
				display("{} {}".format(k,v))
	
	if args.file_output:
		f.close()

else:
	for k,v in cluster_counts.most_common():
		print k, v

# Look at specific clusters interactively
if args.output == 'browser':
	target_cluster = ''
	while target_cluster != 'q':
		cluster_metadata = []
		target_cluster = raw_input("Which cluster do you want to examine? (q to quit):")
		if target_cluster != 'q':
			print ("\tPages in cluster {}:".format(target_cluster))
			
			for c,page in clustered_pages:
				dont_drown_browser = 0
				if c == int(target_cluster) and dont_drown_browser < 10:
					dont_drown_browser+=1
					webpage = page.partition('.')[0]+'.html'
					print webpage

					# Display pages from a given cluster in the broswer (best with few pages!)
					webbrowser.open(page.partition('.')[0]+'.html', new=2) #open in new tab

elif args.output == 'plot':
	# Plot clusters
	plt.scatter(range(1,len(clusters)+1), clusters)
	plt.show()

