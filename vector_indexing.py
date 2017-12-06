"""
Example to how how to index categorical feature columns in a dataset of Vector.
"""

from pyspark.ml.feature import VectorIndexer
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("VectorIndexing")\
        .getOrCreate()

    df = spark.createDataFrame([(Vectors.dense([-1.0, 0.0, 10.0]),),
                                (Vectors.dense([0.0, 1.0, 20.0]), ),
                                (Vectors.dense([0.0, 2.0, 10.0]),)],
                               ["a"])

    df.show()

    indexer = VectorIndexer(maxCategories=3, inputCol="a", outputCol="indexed")
    model = indexer.fit(df)
    print(model.transform(df).collect()[1]["a"])
    print(model.transform(df).collect()[1].indexed)
    print(model.numFeatures)
    print(model.categoryMaps)
    print()
    # Example with bigger data set
    data = spark.read.format("libsvm").load("data/sample_libsvm_data.txt")

    # Create VectorIndexer object with maximum categories set to 10
    indexer = VectorIndexer(inputCol="features",
                            outputCol="indexed",
                            maxCategories=10)
    indexerModel = indexer.fit(data)

    categoricalFeatures = indexerModel.categoryMaps
    print("Chose {} categorical features: {}".format(
        len(categoricalFeatures),
        ", ".join(str(k) for k in categoricalFeatures.keys())))

    # Create new column "indexed" with categorical values transformed to indices
    indexedData = indexerModel.transform(data)
    indexedData.show()

    spark.stop()
