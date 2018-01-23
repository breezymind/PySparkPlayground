"""
Isotonic Regression Example.

isotonic regression or monotonic regression is the technique of fitting a 
free-form line to a sequence of observations under the following constraints: 
    * the fitted free-form line has to be non-decreasing everywhere
    * it has to lie as close to the observations as possible
"""

from pyspark.ml.regression import IsotonicRegression
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("IsotonicRegression")\
        .getOrCreate()

    # Loads data.
    dataset = spark.read.format("libsvm")\
        .load("data/sample_isotonic_regression_libsvm_data.txt")

    # Trains an isotonic regression model.
    model = IsotonicRegression().fit(dataset)
    print("Boundaries in increasing order: {}\n".format(str(model.boundaries)))
    print("Predictions associated with the boundaries: {}\n".format(
        str(model.predictions)))

    # Makes predictions.
    model.transform(dataset).show()

    spark.stop()
