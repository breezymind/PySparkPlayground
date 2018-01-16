"""
Random Forest Classifier Example.

Using libsvm format: <label> <index1:value1> <index2:value2> ... <indexN:valueN>
Data is stored in sparse form, meaning only the non-zero data are stored, and
any missing data is taken as holding value zero.

"""

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("RandomForestClassifier")\
        .getOrCreate()

    # Load and parse the data file, converting it to a DataFrame.
    data = spark.read.format("libsvm").load("data/sample_libsvm_data.txt")

    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol="label",
                                 outputCol="indexedLabel").fit(data)

    # Automatically identify categorical features, and index them.
    # Set maxCategories so features with > 4 distinct values are treated as
    # continuous.
    featureIndexer = VectorIndexer(inputCol="features",
                                   outputCol="indexedFeatures",
                                   maxCategories=4).fit(data)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    # Train a RandomForest model.
    rf = RandomForestClassifier(labelCol="indexedLabel",
                                featuresCol="indexedFeatures",
                                numTrees=10)

    # Convert indexed labels back to original labels.
    labelConverter = IndexToString(inputCol="prediction",
                                   outputCol="predictedLabel",
                                   labels=labelIndexer.labels)

    # Chain indexers and forest in a Pipeline
    pipeline = Pipeline(
        stages=[labelIndexer, featureIndexer, rf, labelConverter])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # model is of type 'pyspark.ml.pipeline.PipelineModel', which represents
    # a compiled pipeline with transformers and fitted models.
    print(type(model))

    # Make predictions.
    predictions = model.transform(testData)  # Returns dataframe

    # Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(10)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel",
                                                  predictionCol="prediction",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = {:.2f}".format(1.0 - accuracy))

    # Get the random forest model (third stage of the pipeline)
    rfModel = model.stages[2]
    print(rfModel)  # summary only

    spark.stop()
