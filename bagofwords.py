import pandas
import os
import numpy
import pprint
import argparse
import matplotlib.pylab as pyp
import matplotlib
import matplotlib.cm as cm
import seaborn as sns
import sklearn
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics
import webbrowser
import subprocess
import collections
import string
import re
import csv
import itertools
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import glob
from sklearn import preprocessing
from sklearn.manifold import TSNE
from wordcloud import WordCloud
import matplotlib.patches as mpatches

# Script takes number of pages and number of clusters as arguments, has optional flags for ways to look at clusters
parser = argparse.ArgumentParser(description='Imports wordcounts from csv and clusters them')
parser.add_argument('wordcount_file', help='csv containing table of wordcounts per page')
#parser.add_argument('num_pages', help='How many webpages to analyze and cluster (max of 500)')
parser.add_argument('num_clusters', help='How many clusters to group pages into')
#parser.add_argument('--page-list', help='Cluster these specific pages, from a .txt file.')
parser.add_argument('--init', default=10, help='Int that serves as seed for cluster generation.')
parser.add_argument('--output', help='Display clusters as [plot], see webpages in [browser], get [sil]houette plot for current cluster number, get [all] silhouette scores and plots for a range of cluster numbers, plot clusters via [tsne] dimensionality reduction, or see [distortion] plot to pick ideal cluster numbers.')
parser.add_argument('--meta', help="Which meta tag to judge clusters by. Interesting options include og:title, Description, Keywords, og:type, CategoryPath, PageType, SalesType.")
parser.add_argument('--file-output', action='store_true', help="Will save cluster details to a file.")
args = parser.parse_args()
args.num_clusters = int(args.num_clusters)
args.init = int(args.init)

'''
# Convert wordcount files to list of dictionaries for each page
if args.page_list:
	wordcounts_f = [f.strip().replace('.html','.w.txt') for f in open(args.page_list,'r')]
	wordcounts_f = wordcounts_f[0:min(int(args.num_pages),len(wordcounts_f))]
else:
	wordcounts_f = [f for f in os.listdir('.') if f.endswith('.w.txt')][0:int(args.num_pages)]
	
wordcounts = []
for fl in wordcounts_f:
	wc_dict = dict()
	for line in open(fl, 'r'):
		count = line.strip().partition(' ')
		if count[2]: # drop empty lines
			wc_dict[count[2]] = int(count[0])
	wordcounts.append(wc_dict)
'''
# stoplist from correlation matrix
redundant_words = ['rating', 'help', 'contacted', 'certification', 'celeron', 'dbc', 'labels', 'gb', 'tools', 'touch', 'japan', 'label', 'previous', 'fibre', 'ads', 'giftcard', 'ships', 'gen', 'environment', 'battery', 'pci', 'choose', 'th', 'combo', 'covered', 'remaining', 'careers', 'finger', 'non', 'return', 'get', 'read', 'preferred', 'primary', 'band', 'datasheets', 'handling', 'rewardterms', 'half', 'front', 'using', 'now', 'emails', 'determines', 'nfc', 'emc', 'customizable', 'taxes', 'standards', 'specific', 'privacy', 'small', 'drivers', 'sheet', 'www', 'set', 'korea', 'generation', 'spring', 'configured', 'webbank', 'phi', 'arrives', 'bluetooth', 'malware', 'fiber', 'us', 'eu', 'techcenter', 'balance', 'specs', 'separately', 'shown', 'rw', 'dense', 'hdmi', 'fgeneric', 'ddr', 'epeat', 'kensington', 'your', 'hca', 'safety', 'market', 'reader', 'got', 'net', 'integrated', 'rj', 'full', 'terms', 'business', 'eco', 'cart', 'hours', 'qualify', 'computrace', 'china', 'french', 'trademarks', 'bose', 'card', 'box', 'solid', 'unresolved', 'license', 'spill', 'adapter', 'compliance', 'egift', 'credit', 'ltd', 'vault', 'cto', 'processor', 'south', 'wlan', 'smartcard', 'pays', 'wigig', 'height', 'spanish', 'included', 'masthead', 'apply', 'tvs', 'total', 'forums', 'minitower', 'select', 'size', 'blog', 'promoterms', 'usb', 'billing', 'capable', 'encryption', 'engagement', 'seamless', 'width', 'camera', 'refurbished', 'low', 'memory', 'pf', 'assessment', 'feedback', 'asset', 'life', 'rewards', 'form', 'hr', 'registered', 'precision', 'lists', 'wi', 'environmental', 'optical', 'representative', 'atom', 'hd', 'applied', 'financing', 'endpoint', 'fips', 'anything', 'mm', 'double', 'mo', 'widescreen', 'enabled', 'value', 'protected', 'arrive', 'pre', 'purchases', 'register', 'premier', 'drive', 'hardware', 'please', 'home', 'close', 'serial', 'uncheck', 'different', 'btn', 'pcie', 'newsroom', 'mic', 'damage', 'click', 'varies', 'internal', 'pad', 'digital', 'sdhc', 'contactless', 'sim', 'map', 'product', 'used', 'diagnosis', 'separate', 'customize', 'price', 'loyalty', 'weee', 'includes', 'wired', 'whr', 'organization', 'sff', 'fi', 'webcam', 'remote', 'framework', 'charges', 'tpm', 'corea', 'lcd', 'wwan', 'savings', 'workspace', 'professional', 'model', 'typically', 'order', 'sd']
flat_words = ['hz', 'en', 'better', 'videos', 'testing', 'number', 'analytics', 'businesses', 'want', 'vostro', 'yes', 'open', 'vmware', 'chromebook', 'vrtx', 'console', 'powered', 'quality', 'mib', 'long', 'increase', 'parts', 'cloud', 'innovation', 'red', 'files', 'users', 'big', 'exchange', 'max', 'processing', 'tested', 'web', 'device', 'unique', 'change', 'psu', 'store', 'rugged', 'faster', 'cable', 'platforms', 'fully', 'blade', 'appliance', 'times', 'large', 'length', 'ultrasharp', 'stand', 'cards', 'os', 'view']
#frequent_words = ['access', 'ads', 'advantage', 'apply', 'atom', 'available', 'back', 'business', 'call', 'celeron', 'chat', 'click', 'code', 'com', 'community', 'company', 'conditions', 'core', 'credit', 'data', 'day', 'days', 'emails', 'features', 'feedback', 'financing', 'form', 'free', 'full', 'get', 'gif', 'help', 'high', 'inside', 'intel', 'issues', 'itanium', 'learn', 'legal', 'logo', 'manage', 'new', 'one', 'online', 'order', 'page', 'payment', 'pentium', 'power', 'privacy', 'products', 'purchase', 'purchases', 'read', 'regulatory', 'rewards', 'sale', 'security', 'separately', 'services', 'shop', 'small', 'statement', 'support', 'system', 'systems', 'technology', 'terms', 'time', 'tm', 'trademarks', 'ultrabook', 'us', 'used', 'using', 'valid', 'vpro', 'work', 'xeon']
bad_words = redundant_words + flat_words# + frequent_words
#bad_words = []

wordcounts = []
wordcounts_f = []
with open(args.wordcount_file,'r') as wordcount_file:
	wordcount_data = [r for r in csv.reader(wordcount_file, delimiter=' ')]
	headers = wordcount_data[0]
	wordcount_data = wordcount_data[1:]
	for page in wordcount_data:
		wc_dict = {headers[i]:count for i,count in enumerate(page)}
		wordcounts_f.append(wc_dict.pop('URL')) # add URL to file name list, and remove from wordcount dict
		for k in bad_words: # stop list
			wc_dict.pop(k,None)
		wordcounts.append(wc_dict)

# Use DictVectorizer to convert the dictionaries to a sparse array for sklearn
word_features = DictVectorizer().fit_transform(wordcounts).toarray()
#word_features = sklearn.preprocessing.normalize(features) # remove variation in size of page

# K-means clustering
raw_clusters = KMeans(n_clusters=int(args.num_clusters), n_init=10, random_state=args.init).fit(word_features)
clusters = raw_clusters.labels_

def display(text):
	# print to screen or prints to file, depending on options set
	if args.file_output:
		f.write(text +'\n')
	else:
		print text

if args.file_output:
	try:
		f = open('../clusters_by_{}.txt'.format(args.meta.translate(string.maketrans("",""), string.punctuation)),'w')
	except:
		f = open('../clusters.txt','w')

'''
wordcloud = WordCloud(background_color='white',width=1200,height=1000).generate(wordcounts[1].keys())
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
'''
def custom_legend(colors,labels, legend_location = 'upper left', legend_boundary = (1,1)):
	# Create custom legend for colors
	recs = []
	for i in range(0,len(colors)):
		recs.append(mpatches.Rectangle((0,0),1,1,fc=colors[i]))
	pyp.legend(recs,labels,loc=legend_location, bbox_to_anchor=legend_boundary)
	
if args.output == 'tsne':
	#tSNE visualization
	RS = 20150101
	wc_proj = TSNE(random_state=RS).fit_transform(word_features)

	palette = numpy.array(sns.color_palette("hls", args.num_clusters))

	# plot it with colors indicating clusters
	f = plt.figure(figsize=(8, 8))
	ax = plt.subplot(aspect='equal')
	sc = ax.scatter(wc_proj[:,0], wc_proj[:,1], lw=0, s=40, c=palette[clusters.astype(numpy.int)])
	custom_legend(palette,xrange(0,args.num_clusters))
	plt.xlim(-25, 25)
	plt.ylim(-25, 25)
	ax.axis('off')
	ax.axis('tight')

	plt.show()


# Display cluster information
cluster_counts = collections.Counter(clusters)
if args.meta:
	display("\tCluster overview:")
	display('Average silhouette score: {}'.format(metrics.silhouette_score(word_features, clusters, metric='euclidean')))

	for k,v in cluster_counts.most_common():
		display("{}: {}".format(k,v))
	
	# load metadata from all pages
	all_metadata = []
	metadata_f = []
	meta_file = args.wordcount_file.partition('wordcounts.csv')[0] +'meta.csv'
	with open(meta_file,'r') as metadata_file:
		metadata_data = [r for r in csv.reader(metadata_file, delimiter=',')]
		headers = metadata_data[0]
		metadata_data = metadata_data[1:]
		for page in metadata_data:
			md_dict = {headers[i]:count for i,count in enumerate(page)}
			metadata_f.append(md_dict.pop('URL')) # add URL to file name list, and remove from metadata dict
			all_metadata.append(md_dict)
	# fill in empty pages that had no metadata
	missing_meta = set(wordcounts_f).difference(metadata_f)
	print "missing: {}".format(len(missing_meta))
	
	meta_table = zip(metadata_f,all_metadata)
	meta_table.sort(key=lambda x: wordcounts_f.index(x[0])) # so metadata is in same order as clusters
	metadata_f, all_metadata = zip(*meta_table)
	
	# get only target metadata
	target_meta = [m[args.meta] for m in all_metadata]
	
	if args.meta == 'CategoryPath': # special format
		#metadata = [w.rpartition('/')[2].split('-') for w in metadata] # grab only last category
		#remove parts of categories that are numeric (to simplify the category space)
		target_meta = [w.partition('/')[2].partition('/')[0].split('-') for w in target_meta] # grab only second category
		target_meta = map(lambda x: "-".join([w for w in x if w.isalpha()]), target_meta)
	
	cluster_metadata = zip(clusters,target_meta)
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
		
		if args.meta == 'CategoryPath': # special format
			#all_words = [item for sublist in [w.split('/') for w in cluster_metadata] for item in sublist] # get full list of categories
			all_words = current_metadata # already collapsed categories after collecting them
		else:
			# count words; drop words with a frequency of 1 or symbols/numbers
			all_words = [w for w in " ".join(current_metadata).translate(string.maketrans("",""), string.punctuation).split(' ') if w.isalpha()]
		bag_words = collections.Counter({k: c for k, c in collections.Counter(all_words).items() if k})

		for k,v in bag_words.most_common():
			if cluster_counts[cl] >1 and v <2 and args.meta in ['og:title', 'og:description', 'Title', 'Description', 'Keywords', 'CategoryPath']: # filter out words that appear only once
				pass;
			else:
				display("{} {}".format(k,v))
	
	# Plot clusters versus metadata
	sorted_labels = sorted(list(set(target_meta)), reverse=True)
	meta_to_int = {v:k for k,v in enumerate(sorted_labels)}
	meta_index = [meta_to_int[m] for m in target_meta]
	pyp.scatter(clusters, meta_index, alpha=.1, s=200) 
	pyp.yticks(meta_to_int.values(), meta_to_int.keys())
	pyp.ylabel(args.meta,rotation=90)
	pyp.ylim([0,len(meta_to_int.keys())])
	pyp.xticks(xrange(0,args.num_clusters))
	pyp.xlabel('Cluster')
	pyp.xlim([-1,args.num_clusters])
	pyp.grid(b=True, axis='y', which='major')
	pyp.show()
	
	if args.file_output:
		f.close()

else:
	display("\tCluster overview:")
	display('Average silhouette score: {}'.format(metrics.silhouette_score(word_features, clusters, metric='euclidean')))

	for k,v in cluster_counts.most_common():
		display("\t{}: {}".format(k,v))
		#for cl,page in clustered_pages:
		#	if cl == k:
		#		display(page)


if args.output == 'distortion':
	# elbow plot to judge ideal cluster num (very slow!)
	meandistortions = []
	K = range(1,args.num_clusters)
	for k in K:
		kmeans = KMeans(n_clusters=k, n_init=10, random_state=args.init).fit(word_features)
		meandistortions.append(sum(numpy.min(cdist(word_features, kmeans.cluster_centers_, 'euclidean'), axis=1)) / word_features.shape[0])
	plt.plot(K, meandistortions, 'bx-')
	plt.xlabel('number of clusters')
	plt.ylabel('Average distortion')
	plt.title('Selecting k with the Elbow Method')
	plt.show()
	
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
		palette = numpy.array(sns.color_palette("hls", args.num_clusters))

		for i in range(args.num_clusters):
			# Aggregate the silhouette scores for samples belonging to
			# cluster i, and sort them
			ith_cluster_silhouette_values = \
				sample_silhouette_values[clusters == i]

			ith_cluster_silhouette_values.sort()

			size_cluster_i = ith_cluster_silhouette_values.shape[0]
			y_upper = y_lower + size_cluster_i

			color = palette[i]#cm.spectral(float(i) / args.num_clusters)
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

		clusterer = KMeans(n_clusters=n_clusters, random_state=args.init)
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
	clustered_pages = zip(clusters,wordcounts_f)
	clustered_pages.sort(key=lambda x: x[0])

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

