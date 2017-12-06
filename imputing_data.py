"""
Example of how to impute missing data, either using the mean or the median 
of the columns in which the missing values are located. The input columns 
should be of DoubleType or FloatType. Currently Imputer does not support 
categorical features and possibly creates incorrect values for a categorical 
feature.

Note that the mean/median value is computed after filtering out missing values.
"""

from pyspark.ml.feature import Imputer
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("ImputingData")\
        .getOrCreate()

    df = spark.createDataFrame([
        (1.0, float("nan")),
        (2.0, float("nan")),
        (float("nan"), 3.0),
        (4.0, 4.0),
        (5.0, 5.0)
    ], ["a", "b"])

    # Creating an Imputer object and fitting the data
    imputer = Imputer(inputCols=["a", "b"], outputCols=["out_a", "out_b"])
    print(imputer.getStrategy())
    model = imputer.fit(df)

    # surrogateDF returns a dataframe containing input columns and their
    # corresponding surrogates, which are used to replace the missing values in
    # the input dataframe.
    model.surrogateDF.show()

    # Transforming the data and displaying it
    model.transform(df).show()

    # Applying different imputation strategy 'median'
    imputer.setStrategy("median").fit(df).transform(df).show()
    print(imputer.getStrategy())

    spark.stop()
