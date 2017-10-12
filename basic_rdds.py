from pyspark import SparkConf, SparkContext

conf = SparkConf()\
    .setMaster("local[*]")\
    .setAppName("Basic RDDs")
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

print("Squares:")
nums = sc.parallelize([1, 2, 3, 4])
squared = nums.map(lambda x: x * x).collect()
for num in squared:
    print(num)
print("**********************")
print("Words:")
lines = sc.parallelize(["hello world", "hi"])
words = lines.flatMap(lambda line: line.split(" "))
print(words.collect())
print("**********************")
print("Union:")
rdd = sc.parallelize([1,2,3])
other = sc.parallelize([3,4,5])
result = rdd.union(other)
print(result.collect())
print("**********************")
print("Sum:")
total = rdd.reduce(lambda x,y: x + y)
print(total)
print("**********************")
print("Aggregation - Tuple (Sum,Count)")
seqOp = (lambda x,y: (x[0] + y, x[1] + 1))
combOp = (lambda x,y: (x[0] + y[0], x[1] + y[1]))
sumCount = rdd.aggregate((0,0), seqOp, combOp)
print(sumCount)
seqOp2 = (lambda x,y: (x[0], x[1] + 1))
combOp2 = (lambda x,y: (x[0] * y[0], x[1] + y[1]))
productCount = rdd.aggregate((1,0), seqOp2, combOp2)
print(productCount)

sc.stop()




