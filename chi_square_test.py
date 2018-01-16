"""
An example for Chi-square hypothesis testing.

It conducts Pearson's independence test for every feature against the label.
For each feature, the (feature, label) pairs are converted into a contingency
matrix for which the chi-squared statistic is computed. All label and feature 
values must be categorical.

The null hypothesis is that the occurrence of the outcomes is statistically 
independent.
"""

from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import ChiSquareTest

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("ChiSquareTest") \
        .getOrCreate()

    # Create dataframe with label and features vector
    data = [(0.0, Vectors.dense(0.5, 10.0)),
            (0.0, Vectors.dense(1.5, 20.0)),
            (1.0, Vectors.dense(1.5, 30.0)),
            (0.0, Vectors.dense(3.5, 30.0)),
            (0.0, Vectors.dense(3.5, 40.0)),
            (1.0, Vectors.dense(3.5, 40.0))]
    df = spark.createDataFrame(data, ["label", "features"])
    df.show(truncate=False)

    r = ChiSquareTest.test(df, "features", "label").head()
    print("pValues: " + str(r.pValues))
    print("degreesOfFreedom: " + str(r.degreesOfFreedom))
    print("statistics: " + str(r.statistics))

    spark.stop()
