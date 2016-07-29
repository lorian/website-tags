

import pandas
import os
import numpy
import pprint
import argparse
import webbrowser
import subprocess
import collections
import string
import re
import csv
import itertools
import glob

import sklearn
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn import preprocessing
from sklearn.manifold import TSNE

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
parser.add_argument('--save-plots', action='store_true', help="When run on server, will save plots rather than display them. Note filenames might not be unique.")
parser.add_argument('--bad-words', default='all', help="Which sets of stop words to remove. Defaults to [all], but can be [redundant],[flat],or [peak], or any pair of the above with no spaces.")
args = parser.parse_args()
args.num_clusters = int(args.num_clusters)
args.init = int(args.init)

if args.save_plots:
	# so it can run on server without display
	import matplotlib as mpl
	mpl.use('Agg')

import matplotlib.pylab as pyp
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import seaborn as sns
	
# stoplist from correlation matrix
redundant_words_2000 = ['rating', 'help', 'contacted', 'certification', 'celeron', 'dbc', 'labels', 'gb', 'tools', 'touch', 'japan', 'label', 'previous', 'fibre', 'ads', 'giftcard', 'ships', 'gen', 'environment', 'battery', 'pci', 'choose', 'th', 'combo', 'covered', 'remaining', 'careers', 'finger', 'non', 'return', 'get', 'read', 'preferred', 'primary', 'band', 'datasheets', 'handling', 'rewardterms', 'half', 'front', 'using', 'now', 'emails', 'determines', 'nfc', 'emc', 'customizable', 'taxes', 'standards', 'specific', 'privacy', 'small', 'drivers', 'sheet', 'www', 'set', 'korea', 'generation', 'spring', 'configured', 'webbank', 'phi', 'arrives', 'bluetooth', 'malware', 'fiber', 'us', 'eu', 'techcenter', 'balance', 'specs', 'separately', 'shown', 'rw', 'dense', 'hdmi', 'fgeneric', 'ddr', 'epeat', 'kensington', 'your', 'hca', 'safety', 'market', 'reader', 'got', 'net', 'integrated', 'rj', 'full', 'terms', 'business', 'eco', 'cart', 'hours', 'qualify', 'computrace', 'china', 'french', 'trademarks', 'bose', 'card', 'box', 'solid', 'unresolved', 'license', 'spill', 'adapter', 'compliance', 'egift', 'credit', 'ltd', 'vault', 'cto', 'processor', 'south', 'wlan', 'smartcard', 'pays', 'wigig', 'height', 'spanish', 'included', 'masthead', 'apply', 'tvs', 'total', 'forums', 'minitower', 'select', 'size', 'blog', 'promoterms', 'usb', 'billing', 'capable', 'encryption', 'engagement', 'seamless', 'width', 'camera', 'refurbished', 'low', 'memory', 'pf', 'assessment', 'feedback', 'asset', 'life', 'rewards', 'form', 'hr', 'registered', 'precision', 'lists', 'wi', 'environmental', 'optical', 'representative', 'atom', 'hd', 'applied', 'financing', 'endpoint', 'fips', 'anything', 'mm', 'double', 'mo', 'widescreen', 'enabled', 'value', 'protected', 'arrive', 'pre', 'purchases', 'register', 'premier', 'drive', 'hardware', 'please', 'home', 'close', 'serial', 'uncheck', 'different', 'btn', 'pcie', 'newsroom', 'mic', 'damage', 'click', 'varies', 'internal', 'pad', 'digital', 'sdhc', 'contactless', 'sim', 'map', 'product', 'used', 'diagnosis', 'separate', 'customize', 'price', 'loyalty', 'weee', 'includes', 'wired', 'whr', 'organization', 'sff', 'fi', 'webcam', 'remote', 'framework', 'charges', 'tpm', 'corea', 'lcd', 'wwan', 'savings', 'workspace', 'professional', 'model', 'typically', 'order', 'sd']
flat_words_2000 = ['vrtx', 'files', 'innovation', 'stand', 'unique', 'increase', 'appliance', 'faster', 'number', 'exchange', 'vmware', 'ultrasharp', 'cards', 'open', 'red', 'testing', 'want', 'cover', 'functionality', 'cable', 'os', 'device', 'blade', 'vostro', 'run', 'today', 'quality', 'density', 'cloud', 'chromebook', 'phone', 'en', 'videos', 'large', 'processing', 'users', 'enables', 'mobile', 'tested', 'optimized', 'key', 'simple', 'times', 'analytics', 'rugged', 'yes', 'requirements', 'change', 'while', 'hz']
peak_words_10000 = ['words','access','ads','advantage','ajax','applied','apply','arrive','atom','back','balance','blog','bose','bt','business','call','careers','celeron','charges','chat','click','close','code','com','community','company','compare','conditions','contracts','core','corporate','corporation','countries','coupons','credit','cs','customers','data','date','day','days','discounts','eligible','email','emails','employee','engagement','events','except','expert','expires','extra','features','feedback','financing','find','form','forums','full','get','gif','greater','high','inc','inside','instead','intel','investors','issues','itanium','item','join','law','learn','legal','limited','live','loader','logo','loyalty','make','manage','map','match','medium','minimum','monthly','must','new','newsroom','offered','one','online','order','outlet','page','paid','partnerdirect','payment','payments','pentium','phi','privacy','products','program','prohibited','promotional','provided','purchase','purchases','qualify','read','refurbished','regulatory','responsibility','rewards','rewardterms','sale','search','separate','separately','services','ship','shop','sign','small','social','statement','student','support','system','systems','taxes','techcenter','technology','terms','time','trademarks','typically','ultrabook','unresolved','us','used','using','valid','vary','via','vpro','webbank','work','xeon','yes']
redundant_words_10000 = ['ck', 'ddp', 'help', 'dbc', 'vary', 'auf', 'prix', 'code', 'personnaliser', 'ga', 'dw', 'go', 'tcg', 'verantwortung', 'ist', 'slot', 'clavier', 'ads', 'la', 'verwalten', 'tzliche', 'smartcard', 'valid', 'choose', 'th', 'factor', 'employee', 'smart', 'ra', 'wirelessa', 'return', 'get', 'ihnen', 'disque', 'da', 'band', 'datasheets', 'peut', 'forums', 'nnen', 'front', 'ma', 'itanium', 'du', 'publicita', 'day', 'sans', 'wenn', 'moire', 'nfc', 'offres', 'emc', 'die', 'vente', 'events', 'french', 'poids', 'payments', 'small', 'commentaires', 'mo', 'timing', 'methods', 'generation', 'spring', 'energy', 'aatre', 'back', 'emails', 'bluetooth', 'ftsbedingungen', 'zahlungsoptionen', 'used', 'ex', 'lieferinformationen', 'zu', 'tva', 'click', 'techniques', 'cliquez', 'disponible', 'livres', 'ddr', 'selected', 'please', 'conomies', 'legal', 'discounts', 'state', 'protected', 'sur', 'gratuite', 'peuvent', 'abbildungen', 'hauteur', 'carte', 'donna', 'emplacement', 'compact', 'korea', 'terms', 'business', 'eco', 'arbeitstagen', 'gif', 'courriers', 'ufe', 'cart', 'wh', 'hours', 'oben', 'les', 'fips', 'free', 'mes', 'qui', 'bose', 'sind', 'pvc', 'certifia', 'zur', 'arrive', 'gescha', 'panier', 'compliance', 'ajax', 'shipping', 'akzeptieren', 'pcie', 'tour', 'chsten', 'hier', 'une', 'lte', 'prohibited', 'konfigurieren', 'verka', 'zum', 'verschickt', 'inta', 'trademarks', 'versand', 'battery', 'rohs', 'zura', 'primary', 'height', 'valables', 'regulatory', 'eine', 'webbank', 'apply', 'total', 'ouvra', 'minitower', 'unit', 'warenkorb', 'usb', 'privacy', 'avec', 'contact', 'width', 'refurbished', 'low', 'statement', 'memory', 'priceagbp', 'zzgl', 'dur', 'vat', 'life', 'rewards', 'lesegera', 'intela', 'offered', 'expresschargea', 'adresse', 'accidental', 'lecteur', 'displaygra', 'environmental', 'part', 'hca', 'produits', 'liste', 'mwst', 'line', 'fil', 'he', 'environnement', 'endpoint', 'conformita', 'bfr', 'gelten', 'produkte', 'des', 'arbeitsspeicher', 'see', 'value', 'angebotspreis', 'careers', 'den', 'adaptateur', 'ber', 'expa', 'balance', 'comparecompare', 'purchases', 'lbs', 'register', 'investors', 'am', 'innerhalb', 'bena', 'gale', 'au', 'vos', 'preise', 'suivant', 'internal', 'unresolved', 'optionen', 'customise', 'jour', 'credit', 'newsroom', 'datenschutz', 'drivers', 'nd', 'rechtliches', 'member', 'juin', 'varier', 'sie', 'gra', 'commande', 'lithium', 'mit', 'erfolgt', 'kunden', 'manuals', 'map', 'product', 'de', 'phi', 'separate', 'may', 'notre', 'price', 'expires', 'includes', 'wired', 'accepting', 'wi', 'times', 'wireless', 'ka', 'modes', 'using', 'tpm', 'zwei', 'drive', 'wwan', 'loader', 'efficient', 'savings', 'optical', 'responsibility', 'south', 'diagnosis', 'basket', 'sa', 'order']
flat_words_10000 = ['yes', 'vostro', 'change', 'blade', 'inspiron', 'interface', 'mobile', 'gigabit', 'fc', 'partnerdirect', 'web', 'bios', 'alienware', 'rugged', 'amd', 'education', 'premium', 'ideal', 'vmware', 'device', 'connect', 'ultra', 'simple', 'space', 'ultrabook', 'details', 'supportassist', 'sc', 'deals', 'big', 'micro', 'base', 'content', 'tape', 'latitude', 'flexible', 'displays', 'open', 'one', 'xps', 'backup', 'class', 'xeona', 'microsoft', 'application', 'ecc', 'users', 'configuration', 'speed', 'multi']

bad_words = []
if args.bad_words == 'all':
	bad_words = redundant_words_10000 + flat_words_10000 + peak_words_10000
if args.bad_words.startswith('redundant') or args.bad_words.endswith('redundant'):
	bad_words += redundant_words_10000
if args.bad_words.startswith('flat') or args.bad_words.endswith('flat'):
	bad_words += flat_words_10000
if args.bad_words.startswith('peak') or args.bad_words.endswith('peak'):
	bad_words += peak_words_10000

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

	if args.save_plots:
		plt.savefig(args.wordcount_file.rpartition('_wordcounts')[0] +"_tsne.png")
	else:
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

