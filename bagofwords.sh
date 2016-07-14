#!/bin/bash
for FILE in *.html
do
	# to avoid "File name too long" errors when we add new extentions:
	FILEBASE=$(basename $FILE .html)
	# strips html from file
	cat $FILE | lynx --stdin --dump > ${FILEBASE}.txt 
	# gets the lines before References (the line number where References is comes from the grep command), 
	# then converts all whitespace to spaces, removes duplicate spaces, removes everything containing a non-letter, inserts line returns between words, 
	# lowercases everything, counts words, then saves output as text file
	head -n $((`grep -nx 'References' ${FILEBASE}.txt | grep -Eo '^[^:]+'` -1)) ${FILEBASE}.txt | tr '[:space:]' ' ' |tr -c '[:alpha:] ' ' ' | tr -s ' '|  tr ' ' '\n' | tr '[:upper:]' '[:lower:]' | sort | uniq -c > ${FILEBASE}.w.txt
<<<<<<< HEAD
done
=======
done
>>>>>>> ec05f1e8c9a128ce3a6294788a6bacc888c060f5
