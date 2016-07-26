# Convert wordcount files to csv
import csv
import os
import collections
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('directory')
args = parser.parse_args()
directory = args.directory

wordcounts_f = [f for f in os.listdir(directory) if f.endswith('.w.txt')]
top_words = [line.strip().split(' ') for line in open(os.path.join(directory,'top_words.txt'),'r')][0]

# file will be <last step of directory name>_wordcounts.csv
with open(directory.split('/')[-2] +'_wordcounts.csv', 'wb') as csvfile:
	countwriter = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
	countwriter.writerow(['URL'] + top_words)
	
	for page in wordcounts_f:
		countdict = collections.defaultdict(int) # defaultdict will pass a 0 if it doesn't have an entry for that key
		for line in open(os.path.join(directory,page), 'r'):
			count = line.strip().partition(' ')
			if count[2] in top_words:
				countdict[count[2]] = int(count[0])
		
		countwriter.writerow([page.partition('_')[0]] + [countdict[word] for word in top_words])

