"""
SparkML utilities for linear algebra. 
For dense vectors, SparkML uses the NumPy array type, so you can simply pass 
NumPy arrays around. 
For sparse vectors, users can construct a SparseVector object from SparkML or 
pass SciPy scipy.sparse column vectors if SciPy is available in their 
environment.
"""

from pyspark.sql import SparkSession
from pyspark.ml.linalg import DenseVector, SparseVector, DenseMatrix
import numpy as np
import array

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("LinearAlgebra")\
        .getOrCreate()

    u = DenseVector([4.0, 6.0])
    v = DenseVector([1.0, 0.0])
    print(u+v)

    # Computing the dot product of two vectors
    dense = DenseVector(array.array('d', [1., 2.]))
    print(dense.dot(dense))
    print(dense.dot(SparseVector(2, [0,1], [2., 1.])))
    print(dense.dot(range(1,3)))
    print(dense.dot(np.array(range(1,3))))

    # Calculating the norm of a DenseVector
    a = DenseVector([0, -1, 2, -3])
    print(a.norm(1))
    print(a.norm(2))

    # Calculating squared distance
    dense1 = DenseVector(array.array('d', [1., 2.]))
    print(dense1.values)
    print(dense1.squared_distance(dense1))
    dense2 = np.array([2., 1.])
    print(dense1.squared_distance(dense2))
    dense3 = [2., 1.]
    print(dense1.squared_distance(dense3))
    sparse1 = SparseVector(2, [0, 1], [2., 1.])
    print(dense1.squared_distance(sparse1))

    # Creating dense matrices
    m = DenseMatrix(2, 2, range(4))  # Column-major dense matrix
    print(m.toArray())

    s = m.toSparse()
    print(s)

    spark.stop()
