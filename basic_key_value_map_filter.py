import sys
from pyspark import SparkContext


def basicKeyValueMapFilter(input):
    """Construct a key/value RDD and then filter on the value"""
    print("Input:")
    print(input.collect())
    # Take first word and add to tuple (first word, full sentence)
    split_input = input.map(lambda x: (x.split(" ")[0],x))
    print("Split input:")
    print(split_input.collect())

    output = input.map(lambda x: (x.split(" ")[0], x)).filter(
        lambda x: len(x[1]) < 20)
    print("Output:")
    print(output.collect())
    return output


if __name__ == "__main__":
    master = "local"
    if len(sys.argv) == 2:
        master = sys.argv[1]
    sc = SparkContext(master, "BasicKeyValueMapFilter")
    input = sc.parallelize(
        ["coffee", "i really like coffee", "coffee > magic", "panda < coffee"])
    output = sorted(basicKeyValueMapFilter(input).collect())
    print(type(output))
    for elem in output:
        print(elem)
