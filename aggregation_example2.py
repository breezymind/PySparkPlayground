from pyspark import SparkConf, SparkContext

conf = SparkConf()\
    .setMaster("local[*]")\
    .setAppName("Aggregation Example 2")
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

inputrdd = sc.parallelize(
    [("maths", 21),
     ("english", 22),
     ("science", 31)],
    3
)

# How many partitions are being used?
print(inputrdd.getNumPartitions())
print(inputrdd.collect())

# seqOp = (lambda value: (value[1],1))
# combOp = (lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1]))
# result = inputrdd.aggregate((3,0), seqOp, combOp)

seqOp = (lambda acc, value: (acc + value[1]))
combOp = (lambda acc1, acc2: (acc1 + acc2))
result = inputrdd.aggregate(3, seqOp, combOp)
print(result)

sc.stop()
