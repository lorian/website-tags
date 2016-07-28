# Create csv containing metadata
import csv
import os
import collections
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('directory')
args = parser.parse_args()
directory = args.directory

webpage_f = [f for f in os.listdir(directory) if f.endswith('.html')]
meta_tags = ['Keywords','og:title','og:description','og:type','CategoryPath','SalesType','Language']
print "Webpages processed: {}".format(len(webpage_f))
# get metadata from all pages

metadata = []
with open(directory.split('/')[-2] +'_meta.csv', 'wb') as csvfile:
	metawriter = csv.writer(csvfile, delimiter=',')
	metawriter.writerow(['URL'] + meta_tags)
	
	for webpage in webpage_f:
		meta_dict = collections.defaultdict(str) # defaultdict will pass a '' if it doesn't have an entry for that key
		# gets all desired meta tags from raw html
		meta = subprocess.Popen("grep '^<meta' {} | grep -e '{}'".format(os.path.join(directory,webpage),'\' -e \''.join(meta_tags)), shell=True, stdout=subprocess.PIPE)
		# gets only content, removes "Dell" tag on the end of some, then removes quotes and spaces at ends
		meta_dict.update({m.partition('="')[2].partition('"')[0]: m.partition('content=')[2].partition('/>')[0].partition('| Dell')[0].strip().strip('"').strip().lower() for m in meta.communicate()[0].split('\n')})
		meta_dict.pop('')
		
		metawriter.writerow([webpage] + [meta_dict[m] for m in meta_tags]) # uses full page name
