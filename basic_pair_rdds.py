from pyspark import SparkConf, SparkContext
import os

conf = SparkConf().setMaster("local[*]").setAppName("Pair RDDs")
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

currentDirectory = os.getcwd()
inputFile = "file://" + currentDirectory + "/data/README.md"

inputRDD = sc.textFile(inputFile)

# Creating a pairRDD using the first word as the key
pairsRDD = inputRDD.map(lambda x: (x.split(" ")[0], x))
print(pairsRDD.take(5))

# Transformations on pairRDDs
rdd = sc.parallelize([(1,2), (3,4), (3,6)])
gbk = rdd.groupByKey()
for k,v in gbk.collect():
    print(k, "-->", list(v))

mv = rdd.mapValues(lambda x: x + 10)
for k, v in mv.collect():
    print(k, "-->", v)

other = sc.parallelize([(3,9)])
joined = rdd.join(other)
for k,v in joined.collect():
    print(k, "-->", list(v))

# Simple filter on length of key
resultRDD = pairsRDD.filter(lambda x: len(x[0]) > 0)
print(resultRDD.take(5))

sc.stop()




