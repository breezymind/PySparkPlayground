from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local[*]").setAppName("Aggregations")
# conf = SparkConf().setMaster("spark://Dirks-iMac.local:7077").setAppName("Aggregations")
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

data = [("panda", 0), ("pink", 3), ("pirate", 3), ("panda", 1), ("pink", 4)]

# Per-key average with reduceByKey() and mapValues()
inputRDD = sc.parallelize(data)
result = inputRDD.mapValues(lambda x: (x,1)).reduceByKey(lambda x,y: (x[0] + y[0], x[1] + y[1]))
for k,v in result.collect():
    print(k, "-->", v)
print("-------------------------")

# Per-key average using combineByKey()
numsRDD = sc.parallelize([(1,2.), (1,4.), (2,10.), (3,1.), (3,2.), (3,4.)])
sumCount = numsRDD\
    .combineByKey((lambda x: (x,1)),
                  (lambda x, y: (x[0] + y, x[1] + 1)),
                  (lambda x, y: (x[0] + y[0], x[1] + y[1])))\
    .map(lambda x: (x[0],x[1][0]/x[1][1]))\
    .collectAsMap()

print(sumCount)

sc.stop()
