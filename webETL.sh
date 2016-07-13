#!/bin/bash
for FILE in *.html
do
	# strips html
	cat $FILE | lynx --stdin --dump | sed -n '/References$/q;p'| tr '[:space:]' ' ' | tr -c '[:alpha:]' ' ' | tr -s ' ' | tr ' ' '\n' | tr '[:upper:]' '[:lower:]' | sort | uniq -c | > ${FILE:0:7}_count.txt
done