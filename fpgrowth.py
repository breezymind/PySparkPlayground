"""
An example demonstrating FPGrowth (Frequent Patterns).

Given a dataset of transactions, the first step of FP-growth is to calculate 
item frequencies and identify frequent items. The second step of FP-growth 
uses a suffix tree (FP-tree) structure to encode transactions without generating 
candidate sets explicitly, which are usually expensive to generate. 
After the second step, the frequent itemsets can be extracted from the FP-tree.

Two important parameters:
    @:param minSupport: the minimum support for an itemset to be identified as 
    frequent.
    @:param minConfidence: minimum confidence for generating Association Rule.
"""

from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("FPGrowth")\
        .getOrCreate()

    df = spark.createDataFrame([
        (0, [1, 2, 5]),
        (1, [1, 2, 3, 5]),
        (2, [1, 2])
    ], ["id", "items"])

    df.show()

    fpGrowth = FPGrowth(itemsCol="items", minSupport=0.5, minConfidence=0.6)
    model = fpGrowth.fit(df)

    # Display frequent itemsets.
    model.freqItemsets.show()

    # Display generated association rules.
    model.associationRules.show()

    # transform examines the input items against all the association rules and
    # summarize the consequents as prediction
    model.transform(df).show()

    spark.stop()
