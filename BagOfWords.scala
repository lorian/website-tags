// quick and dirty word count that includes html tags
// run via spark-shell -i BagOfWords.scala
val textFile = sc.textFile("cdips_dell/000139__http___www_dell_com_us_business_p_networking_products.html")
val counts = textFile.flatMap(line => line.split(" ")).flatMap(line => line.split("<")).flatMap(line => line.split(">")).flatMap(line => line.split("\t")).map(word => (word, 1)).reduceByKey(_ + _)
counts.saveAsTextFile("bagofwords")
System.exit(0)
