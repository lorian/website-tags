#!/bin/bash
for FILE in *.html
do
	# strips html
	cat $FILE | lynx --stdin --dump | sed -n '/^References$/q;p'| tr '[:space:]' ' ' | tr -c '[:alpha:]' ' ' | tr -s ' ' | tr '[:upper:]' '[:lower:]' > ./tmp/${FILE:0:6}_.txt
done