"""
Quantile Discretizer

QuantileDiscretizer takes a column with continuous features and outputs a column 
with binned categorical features. The number of bins can be set using the 
numBuckets parameter. It is possible that the number of buckets used will be 
less than this value, for example, if there are too few distinct values of the 
input to create enough distinct quantiles.
"""

from pyspark.ml.feature import QuantileDiscretizer
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("QuantileDiscretizer")\
        .getOrCreate()

    data = [(0, 18.0), (1, 19.0), (2, 8.0), (3, 5.0), (4, 2.2)]
    df = spark.createDataFrame(data, ["id", "hour"])

    # Output of QuantileDiscretizer for such small datasets can depend on the
    # number of partitions. Here we force a single partition to ensure
    # consistent results.
    # Note this is not necessary for normal use cases
    df = df.repartition(1)

    discretizer = QuantileDiscretizer(numBuckets=3,
                                      inputCol="hour",
                                      outputCol="result")

    result = discretizer.fit(df).transform(df)
    result.show()

    spark.stop()
