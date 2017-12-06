""""
Example to show how to tokenize data using a Tokenizer and a RegexTokenizer. 

A Tokenizer converts the input string to lowercase and then splits it by white 
spaces.

A RegexTokenizer is a regex based tokenizer that extracts tokens eight by using
the provided regex pattern (in Java dialect) to split the text (default) or 
repeatedly matching the regex (if gaps=False). Optional parameters also allow 
filtering tokens using a minimal length.
"""

from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("TokenizingData")\
        .getOrCreate()

    sentenceDataFrame = spark.createDataFrame([
        (0, "Hi I heard about Spark"),
        (1, "I wish Java could use case classes"),
        (2, "Logistic,regression,models,are,neat")
    ], ["id", "sentence"])

    tokenizer = Tokenizer(inputCol="sentence",
                          outputCol="words")

    regexTokenizer = RegexTokenizer(inputCol="sentence",
                                    outputCol="words",
                                    pattern="\\W")
    # alternatively, pattern="\\w+", gaps(False)

    # user-defined function for counting tokens
    countTokens = udf(lambda words: len(words), IntegerType())

    tokenized = tokenizer.transform(sentenceDataFrame)
    tokenized.select("sentence", "words") \
        .withColumn("tokens", countTokens(col("words"))).show(truncate=False)

    regexTokenized = regexTokenizer.transform(sentenceDataFrame)
    regexTokenized.select("sentence", "words") \
        .withColumn("tokens", countTokens(col("words"))).show(truncate=False)

    spark.stop()
