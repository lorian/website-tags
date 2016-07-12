// quick and dirty word count after html tabs have been stripped
// cat 000139__http___www_dell_com_us_business_p_networking_products.html | lynx --stdin --dump > 000139__http___www_dell_com_us_business_p_networking_products.txt 
// run via spark-shell -i BagOfWords.scala
val textFile = sc.textFile("000139__http___www_dell_com_us_business_p_networking_products.txt")
val counts = textFile.map(line => line.replaceAll("""<(?!\/?a(?=>|\s.*>))\/?.*?>""", "")).flatMap(line => line.split(" ")).flatMap(line => line.split("\t")).map(word => (word, 1)).reduceByKey(_ + _)
counts.saveAsTextFile("bagofwords")
System.exit(0)
