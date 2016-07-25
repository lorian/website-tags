#!/bin/bash
# run in directory with .html files, as 'bash ../bagofwords.sh' in default github structure
for FILE in *.html
do
	# to avoid "File name too long" errors when we add new extentions:
	FILEBASE=$(basename $FILE .html)
	# strips html from file
	cat $FILE | lynx --stdin --dump > ${FILEBASE}.txt 
	# gets the lines before References (the line number where References is comes from the grep command), 
	# then converts all whitespace to spaces, removes duplicate spaces, removes everything containing a non-letter, inserts line returns between words, 
	# lowercases everything, removes stop words, counts words, then saves output as text file
	head -n $((`grep -nx 'References' ${FILEBASE}.txt | grep -Eo '^[^:]+'` -1)) ${FILEBASE}.txt | tr '[:space:]' ' ' |tr -c '[:alpha:] ' ' ' | tr -s ' '|  tr ' ' '\n' | tr '[:upper:]' '[:lower:]' | sort | uniq -c | grep -vf ../minimal-stop.txt > ${FILEBASE}.w.txt
done
# combine count files and take top 1000
awk '{ count[$2] += $1 } END { for(elem in count) print count[elem], elem }' *.w.txt | sort -nr | head -n 1000 | tr -d '[:digit:]' | tr -d ' ' > top_words.txt
# drop single-occurance words
#grep -v "^1 " wordcount.txt > wordcount_clean.txt
