"""
Linear Support Vector Machine

A support vector machine constructs a hyperplane or set of hyperplanes in a 
high- or infinite-dimensional space, which can be used for classification, 
regression, or other tasks. Intuitively, a good separation is achieved by the 
hyperplane that has the largest distance to the nearest training-data points of 
any class (so-called functional margin), since in general the larger the margin 
the lower the generalization error of the classifier. 

LinearSVC in Spark ML supports binary classification with linear SVM. 
Internally, it optimizes the Hinge Loss using OWLQN optimizer.
"""

from pyspark.ml.classification import LinearSVC
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("linearSVC")\
        .getOrCreate()

    # Load training data
    training = spark.read.format("libsvm").load("data/sample_libsvm_data.txt")

    lsvc = LinearSVC(maxIter=10, regParam=0.1)

    # Fit the model
    lsvcModel = lsvc.fit(training)

    # Print the coefficients and intercept for linearsSVC
    print("Coefficients: " + str(lsvcModel.coefficients))
    print("Intercept: " + str(lsvcModel.intercept))

    spark.stop()
