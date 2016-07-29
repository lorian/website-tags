#!/bin/bash
for i in *.html
do
	cat $i |grep meta|grep -iw 'og:type\|salestype\|CategoryPath\|Country\|language'|grep -o '".*"'|sed -n '/http/q;p'|sed 's/content=//'|sed 's/ .*all-products\// /'|sed 's/\/.*"//'|sed 's/"//g'|sed 's/-n-workstations//'|sed -n '/description/q;p'>>./tmp/${i:0:6}_header.txt
done