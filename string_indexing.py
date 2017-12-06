"""
Example using StringIndexer and IndexToString

A StringIndex is a label indexer that maps a string column of labels to an ML 
column of label indices.

IndexToString is a transformer that maps a column of indices back to a new 
column of corresponding string values.
"""

from pyspark.ml.feature import IndexToString, StringIndexer
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("StringIndexing")\
        .getOrCreate()

    df = spark.createDataFrame(
        [(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c")],
        ["id", "category"])

    df.show()

    indexer = StringIndexer(inputCol="category",
                            outputCol="categoryIndex")
    model = indexer.fit(df)
    indexed = model.transform(df)

    print("Transformed string column '{}' to indexed column '{}'".format(
          indexer.getInputCol(), indexer.getOutputCol()))
    indexed.show()

    print("StringIndexer will store labels in output column metadata\n")

    converter = IndexToString(inputCol="categoryIndex",
                              outputCol="originalCategory")
    converted = converter.transform(indexed)

    print("Transformed indexed column '{}' back to original string column '{}' "
          "using labels in metadata".format(converter.getInputCol(),
                                            converter.getOutputCol()))
    converted.select("id", "categoryIndex", "originalCategory").show()

    spark.stop()
