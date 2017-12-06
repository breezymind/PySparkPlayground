"""
Simple example of how to use DataFrame for ML.
"""

import os
import tempfile
import shutil
from pyspark.sql import SparkSession
from pyspark.mllib.stat import Statistics
from pyspark.mllib.util import MLUtils

if __name__ == "__main__":

    input = "data/sample_libsvm_data.txt"

    spark = SparkSession \
        .builder \
        .appName("DataFrameExample") \
        .getOrCreate()

    # Loading input data and caching it
    print("Loading LIBSVM file with UDT from " + input + ".")
    df = spark.read.format("libsvm").load(input).cache()
    print("Loaded " + str(df.count()) + " records training data in dataframe.")

    # Printing the schema
    print("Schema from LIBSVM:")
    df.printSchema()

    # Statistical summary of labels.
    labelSummary = df.describe("label")
    labelSummary.show()

    # Convert features columns to new vector type.
    features = MLUtils.convertVectorColumnsFromML(df, "features") \
        .select("features").rdd.map(lambda r: r.features)

    # Printing feature column with average values
    summary = Statistics.colStats(features)
    print("Selected features column with average values:\n" +
          str(summary.mean()))

    # Save the records in a temporary parquet file.
    tempdir = tempfile.NamedTemporaryFile(delete=False).name
    os.unlink(tempdir)
    print("Saving to " + tempdir + " as Parquet file.")
    df.write.parquet(tempdir)

    # Read the records back in.
    print("Loading Parquet file with UDT from " + tempdir)
    newDF = spark.read.parquet(tempdir)
    print("Schema from Parquet:")
    newDF.printSchema()

    # Cleaning up the temporary parquet file
    shutil.rmtree(tempdir)

    spark.stop()
