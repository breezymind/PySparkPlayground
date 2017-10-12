from pyspark import SparkConf, SparkContext

conf = SparkConf()\
    .setMaster("local[*]")\
    .setAppName("Aggregation Example")
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

# Aggregate lets you take an RDD and generate a single value that is of a
# different type than what was stored in the original RDD.

# It does this with three parameters:
# A zeroValue (or initial value) in the format of the result.
# A seqOp function that, given the resulting type and an individual element in
# the RDD, will merge the RDD element into the resulting object.
# The combOb merges two resulting objects together.

people = []
people.append({'name':'Bob', 'age':45,'gender':'M'})
people.append({'name':'Gloria', 'age':43,'gender':'F'})
people.append({'name':'Albert', 'age':28,'gender':'M'})
people.append({'name':'Laura', 'age':33,'gender':'F'})
people.append({'name':'Simone', 'age':18,'gender':'T'})

peopleRdd=sc.parallelize(people)

print(peopleRdd.take(2))            # returns list with 2 elements
# print(len(peopleRdd.collect()))     # returns number of elements in RDD
print(peopleRdd.count())            # returns number of elements in RDD

# We want to take a list of records about people and then we want to sum up
# their ages and count them. So for this example the type in the RDD will be
# a dictionary in the format of {name: NAME, age:AGE, gender:GENDER}.
# The result type will be a tuple that looks like so (Sum of Ages, Count)

seqOp = (lambda x, y: (x[0] + y['age'], x[1] + 1))
combOp = (lambda x, y: (x[0] + y[0], x[1] + y[1]))

result = peopleRdd.aggregate((0,0), seqOp, combOp)
print(result)

# The combOp seems unnecessary but in the map reduce world of spark you need
# that separate operation. Realize that these functions are going to be
# parallelized. peopleRDD is partitioned up. And dependending on its source and
# method of converting the data to an RDD each row could be on its own
# partition.

# partition - A partition is how the RDD is split up. If our RDD was 100,000
# records we could have as many as 100,000 partitions or only 1 partition
# depending on how we created the RDD.

# task - A small job that operates on a single partition. A single task can run
# on only one machine at a time and can operate on only one partiton at a time.

# For the aggregate function the seqOp will run once for every record in a
# partition. This will result in a resulting object for each partition.
# The combOp will be used to merge all the resulting objects together.

females = peopleRdd.filter(lambda x: x['gender'] == 'F')
print(females.collect())
# females.saveAsTextFile('females.txt')  # Files gets written to HDFS by default
# females.saveAsTextFile('file:///tmp/females.txt')
males = peopleRdd.filter(lambda x: x['gender'] == 'M')
print(males.collect())

# import time           # Added to give me time to check the DAG
# time.sleep(60)

sc.stop()
