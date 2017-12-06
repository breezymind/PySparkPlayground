import sys
from pyspark import SparkContext

def basicAvg(nums):
    """Compute the avg"""
    # Fold() is similar to reduce, but takes an initial value
    sumCount = nums.map(lambda x: (x, 1)).fold(
        (0, 0), (lambda x, y: (x[0] + y[0], x[1] + y[1])))
    return sumCount[0] / float(sumCount[1])

def basicProduct(nums):
    """Compute the product of the elements"""
    product = nums.map(lambda x: (x,1)).fold(
        (1,0), (lambda x, y: (x[0] * y[0], x[1] + y[1])))
    return product


if __name__ == "__main__":
    # master = "yarn"
    master = "local[2]"
    if len(sys.argv) == 2:
        master = sys.argv[1]
    sc = SparkContext(master, "BasicAvg")

    nums = sc.parallelize([1, 2, 3, 4])
    avg = basicAvg(nums)
    print("Average:", avg)
    prod = basicProduct(nums)
    print("Product:", prod[0])
    print("Number of elements:", prod[1])
