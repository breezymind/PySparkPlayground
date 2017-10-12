import sys
import tokenize

from pyspark import SparkContext, SparkConf

def basicSquare(nums):
    """Square the numbers"""
    return nums.map(lambda x: x * x)

if __name__ == "__main__":

    conf = SparkConf() \
        .setMaster("local[*]") \
        .setAppName("Basic FlatMap Example")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    # Sometimes we want to produce multiple output elements for each input
    # element. The operation to do this is called flatMap().
    # As with map(), the function we provide to flatMap() is called individually
    # for each element in our input RDD. Instead of returning a single element,
    # we return an iterator with our return values. Rather than producing an RDD
    # of iterators, we get back an RDD that consists of the elements from all
    # of the iterators.
    lines = sc.parallelize(["hello world", "hi"])
    words = lines.flatMap(lambda line: line.split(" "))
    print(words.first())  # returns "hello"

    tokens = lines.map(lambda line: line)
    print(tokens.collect())
    print(tokens.first())

