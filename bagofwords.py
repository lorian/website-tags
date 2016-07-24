# run in directory with wordcount files ending in .w.txt (created by bagofwords.sh)

import pandas
import os
import numpy
import pprint
import argparse
import matplotlib.pylab as pyp
import matplotlib
import matplotlib.cm as cm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics
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
parser.add_argument('--output', help='Display clusters as [plot], see webpages in [browser], get [sil]houette plot for current cluster number, or get [all] silhouette scores and plots for a range of cluster numbers.')
parser.add_argument('--meta', help="Which meta tag to judge clusters by. Interesting options include og:title, Description, Keywords, og:type, CategoryPath, PageType, SalesType.")
parser.add_argument('--file-output', action='store_true', help="Flag will output cluster details to a file. Requires --meta to be set.")
args = parser.parse_args()
args.num_clusters = int(args.num_clusters)

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
raw_clusters = KMeans(n_clusters=int(args.num_clusters), n_init=10).fit(word_features)
clusters = raw_clusters.labels_
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
	
	# get metadata from all pages
	metadata = []
	for f in wordcounts_f:
		webpage = f.partition('.')[0]+'.html'
		# List page content from metadata
		title = subprocess.Popen('grep "{}" {}'.format(args.meta, webpage), shell=True, stdout=subprocess.PIPE)
		# gets only content, removes "Dell" tag on the end of some, then removes quotes and spaces at ends
		metadata.append(title.communicate()[0].partition('content=')[2].partition('/>')[0].partition('| Dell')[0].strip().strip('"').strip().lower())

	cluster_metadata = zip(clusters,metadata)
	for cl in cluster_counts.keys():
		current_metadata = []
		display("\n\tCluster {} ({} pages):".format(cl,cluster_counts[cl]))

		for c,data in cluster_metadata:
			if cl == int(c):
				current_metadata.append(data)
		# check how many pages had metadata of this type
		has_content = collections.Counter(current_metadata)
		if has_content['']:
			display("\tPages without this metadata: {}".format(has_content['']))
		
		# count words; drop words with a frequency of 1 or symbols/numbers
		if args.meta == 'CategoryPath': # special format
			#all_words = [item for sublist in [w.split('/') for w in cluster_metadata] for item in sublist] # get full list of categories
			all_words = [w.rpartition('/')[2].split('-') for w in current_metadata] # grab only last category
			#remove parts of categories that are numeric (to simplify the category space)
			all_words = map(lambda x: "-".join([w for w in x if w.isalpha()]), all_words)
		else:
			all_words = [w for w in " ".join(current_metadata).translate(string.maketrans("",""), string.punctuation).split(' ') if w.isalpha()]
		bag_words = collections.Counter({k: c for k, c in collections.Counter(all_words).items() if k})

		for k,v in bag_words.most_common():
			if cluster_counts[cl] >1 and v <2 and args.meta in ['Title', 'Description', 'Keywords', 'CategoryPath']: # filter out words that appear only once
				pass;
			else:
				display("{} {}".format(k,v))
		
	# Plot clusters versus metadata
	meta_to_int = {v:k for k,v in enumerate(set(metadata))}
	meta_int = [meta_to_int[m] for m in metadata]
	pyp.scatter(clusters, meta_int) 
	pyp.yticks(meta_to_int.values(), meta_to_int.keys())
	pyp.ylabel(args.meta,rotation=90)
	pyp.ylim([0,len(meta_to_int.keys())])
	pyp.xticks(xrange(0,args.num_clusters))
	pyp.xlabel('Cluster')
	pyp.xlim([-1,args.num_clusters])
	pyp.show()
	
	if args.file_output:
		f.close()

else:
	for k,v in cluster_counts.most_common():
		print k, v
	
	print 'Average silhouette score: {}'.format(metrics.silhouette_score(word_features, clusters, metric='euclidean'))
	order_centroids = raw_clusters.cluster_centers_.argsort()[:, ::-1]
	print order_centroids
	
if args.output == 'sil':
		fig, ax1 = pyp.subplots(1, 1)

		ax1.set_xlim([-.3, 1])
		# The (n_clusters+1)*10 is for inserting blank space between silhouette
		# plots of individual cluster_labels, to demarcate them clearly.
		ax1.set_ylim([0, len(word_features) + (args.num_clusters + 1) * 10])

		# The silhouette_score gives the average value for all the samples.
		# This gives a perspective into the density and separation of the formed
		# cluster_labels
		silhouette_avg = metrics.silhouette_score(word_features, clusters)
		print "For {} cluster_labels, average silhouette score is {}".format(args.num_clusters, silhouette_avg)

		# Compute the silhouette scores for each sample
		sample_silhouette_values = metrics.silhouette_samples(word_features, clusters)

		y_lower = 10
		for i in range(args.num_clusters):
			# Aggregate the silhouette scores for samples belonging to
			# cluster i, and sort them
			ith_cluster_silhouette_values = \
				sample_silhouette_values[clusters == i]

			ith_cluster_silhouette_values.sort()

			size_cluster_i = ith_cluster_silhouette_values.shape[0]
			y_upper = y_lower + size_cluster_i

			color = cm.spectral(float(i) / args.num_clusters)
			ax1.fill_betweenx(numpy.arange(y_lower, y_upper),
							  0, ith_cluster_silhouette_values,
							  facecolor=color, edgecolor=color, alpha=0.7)

			# Label the silhouette plots with their cluster numbers at the middle
			ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

			# Compute the new y_lower for next plot
			y_lower = y_upper + 10  # 10 for the 0 samples

		ax1.set_title("The silhouette plot for {} clusters".format(args.num_clusters))
		ax1.set_xlabel("Silhouette coefficient values")
		ax1.set_ylabel("Cluster")

		# The vertical line for average silhoutte score of all the values
		ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

		ax1.set_yticks([])  # Clear the yaxis labels / ticks
		pyp.show()

elif args.output == 'all':
	# Show silhouette scores and plots for clusters +/- 5 around the target number given:
	for n_clusters in xrange(args.num_clusters-5,args.num_clusters+5):
		# Create a subplot
		fig, ax1 = pyp.subplots(1, 1)

		ax1.set_xlim([-.3, 1])
		# The (n_clusters+1)*10 is for inserting blank space between silhouette
		# plots of individual cluster_labels, to demarcate them clearly.
		ax1.set_ylim([0, len(word_features) + (n_clusters + 1) * 10])

		clusterer = KMeans(n_clusters=n_clusters, random_state=10)
		cluster_labels = clusterer.fit_predict(word_features)
		# The silhouette_score gives the average value for all the samples.
		# This gives a perspective into the density and separation of the formed
		# cluster_labels
		silhouette_avg = metrics.silhouette_score(word_features, cluster_labels)
		print "For {} cluster_labels, average silhouette score is {}".format(n_clusters, silhouette_avg)

		# Compute the silhouette scores for each sample
		sample_silhouette_values = metrics.silhouette_samples(word_features, cluster_labels)

		y_lower = 10
		for i in range(n_clusters):
			# Aggregate the silhouette scores for samples belonging to
			# cluster i, and sort them
			ith_cluster_silhouette_values = \
				sample_silhouette_values[cluster_labels == i]

			ith_cluster_silhouette_values.sort()

			size_cluster_i = ith_cluster_silhouette_values.shape[0]
			y_upper = y_lower + size_cluster_i

			color = cm.spectral(float(i) / n_clusters)
			ax1.fill_betweenx(numpy.arange(y_lower, y_upper),
							  0, ith_cluster_silhouette_values,
							  facecolor=color, edgecolor=color, alpha=0.7)

			# Label the silhouette plots with their cluster numbers at the middle
			ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

			# Compute the new y_lower for next plot
			y_lower = y_upper + 10  # 10 for the 0 samples

		ax1.set_title("The silhouette plot for {} clusters".format(n_clusters))
		ax1.set_xlabel("Silhouette coefficient values")
		ax1.set_ylabel("Cluster")

		# The vertical line for average silhoutte score of all the values
		ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

		ax1.set_yticks([])  # Clear the yaxis labels / ticks
		pyp.show()

# Look at specific clusters interactively
elif args.output == 'browser':
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
	pyp.scatter(range(1,len(clusters)+1), clusters)
	pyp.show()

