from pyspark.ml.feature import ElementwiseProduct
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("ElementwiseProduct")\
        .getOrCreate()

    # Create some vector data; also works for sparse vectors
    data = [(Vectors.dense([1.0, 2.0, 3.0]),),
            (Vectors.dense([4.0, 5.0, 6.0]),)]
    df = spark.createDataFrame(data, ["vector"])
    df.show()

    # Transforming data by elementwise product
    transformer = ElementwiseProduct(scalingVec=Vectors.dense([0.0, 1.0, 2.0]),
                                     inputCol="vector",
                                     outputCol="transformedVector")

    # Batch transform the vectors to create new column:
    transformer.transform(df).show()

    spark.stop()
