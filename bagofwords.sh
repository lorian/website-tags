#!/bin/bash
# run from base website-tags $DIRectory
# takes directory containing html files as argument: 'bash bagofwords.sh cdips_dell/'
# must include final /
DIR="$1"
for FILE in ${DIR}*.html
do
	# to avoid "File name too long" errors when we add new extentions:
	FILEBASE=$(basename $FILE .html)
	# strips html from file
	cat $FILE | lynx --stdin --dump > ${DIR}${FILEBASE}.txt 
	# gets the lines before References (the line number where References is comes from the grep command), 
	# then converts all whitespace to spaces, removes duplicate spaces, removes everything containing a non-letter, inserts line returns between words, 
	# lowercases everything, removes stop words, counts words, then saves output as text file
	head -n $((`grep -nx 'References' ${DIR}${FILEBASE}.txt | grep -Eo '^[^:]+'` -1)) ${DIR}${FILEBASE}.txt | tr '[:space:]' ' ' |tr -c '[:alpha:] ' ' ' | tr -s ' '|  tr ' ' '\n' | tr '[:upper:]' '[:lower:]' | sort | uniq -c | grep -vf minimal-stop.txt > ${DIR}${FILEBASE}.w.txt
done
# to avoid argument length issues:
find $DIR -name *.w.txt -exec cat {}  \; > combined_wordcounts.txt
# combine count files and take top 1000 words; remove counts so file is just a single-line list of words
awk '{ count[$2] += $1 } END { for(elem in count) print count[elem], elem }' combined_wordcounts.txt | sort -nr | head -n 1000 | tr -d '[:digit:]' | tr -d ' ' | tr '\n' ' ' > ${DIR}top_words.txt
# convert wordcounts to single csv with top words
python process_bagofwords.py $DIR
