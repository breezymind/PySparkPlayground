"""
Example to show how to remove stopwords from a corpus of text.

Note: null values from input array are preserved unless adding null to stopWords 
explicitly.
"""

from pyspark.ml.feature import StopWordsRemover
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("RemovingStopwords")\
        .getOrCreate()

    sentenceData = spark.createDataFrame([
        (0, ["I", "saw", "the", "red", "balloon"]),
        (1, ["Mary", "had", "a", "little", "lamb"])
    ], ["id", "raw"])

    remover = StopWordsRemover(inputCol="raw", outputCol="filtered")
    remover.transform(sentenceData).show(truncate=False)

    spark.stop()
