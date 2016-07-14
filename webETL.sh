#!/bin/bash
for FILE in *.html
do
	# strips html
	cat $FILE | lynx --stdin --dump | sed -n '/References$/q;p'| tr '[:space:]' ' ' | tr -c '[:alpha:]' ' ' | tr -s ' ' | tr '[:upper:]' '[:lower:]' | sort > ${FILE:0:7}_.txt
done