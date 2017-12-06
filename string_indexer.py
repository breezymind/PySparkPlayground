"""
Simple example of how to use a StringIndexer to transform categories into
category indexes.
"""

from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("StringIndexer")\
        .getOrCreate()

    # Create simple data frame
    df = spark.createDataFrame(
        [(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c")],
        ["id", "category"])

    df.show()

    indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
    indexed = indexer.fit(df).transform(df)
    indexed.show()

    spark.stop()
