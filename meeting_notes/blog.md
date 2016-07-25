
# Sun Jul 24 19:06:46 PDT 2016

Henrik:
supervised learning. got categorise  from ‘CategoryPath’ meta-tag and if it’s empty, use og:type
classification, feature importance
tried tfidf and porter stemmer on and off.



ideas:
extract all og
dimensionality reduction and cluster visualisation with tsne


for Kim:
looked at the pages that have a high count of gif and see if you can see the difference between different categories

# Tue Jul 19 18:29:17 PDT 2016


Kim got bag of words, took out stop words.

Lorian: bow, kmeans, looked at web pages. 500. 


## future ideas:

### look at some websites that are closest to the centroids 
- kmeans has a call for that
- output the website as text
- do it for larger number -- around 500
- make features out of meta tags (for example pagetype feature would be product)
- make a table where feature is the name of a meta and the value is the content

### scaling up 
- remove error and redirects
- adjust on small set and once in a while run on a big set.
- cluster -- split into smaller dataset and run for them

### cleaning bag of words
- statistically remove words that show up in most of the pages
- plot cumulative distribution. x-axis: how many times the word appears in the doc vs y-axis: how many docs with that count of words
- stop dictionary (done already_)
- nltk stemming
- nltk pos. for example, just use nouns 
- use or not use top 10000 or some other number of mos common english words
- (may be ) combine words that have close word vectors

### look at the structure of html
- based on html tag tree structure
- look at pages that have similar links
- look at the count of tags 
- make a feature out of heading, description, title 



