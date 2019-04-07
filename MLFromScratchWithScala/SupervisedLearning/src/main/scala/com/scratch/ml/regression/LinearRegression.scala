package com.scratch.ml.regression

import breeze.linalg.{DenseMatrix, dim, pinv}

class LinearRegression(nIterations: Int = 100, learningRate: Double = 0.001, gradient_descent: Boolean = true)
  extends Regression(nIterations, learningRate) {
  var regularization = 0

  override def fit(X: DenseMatrix[Double], y: DenseMatrix[Double]): Unit = {
    if (gradient_descent) {
      super.fit(X, y)
    } else {
      val x = DenseMatrix.horzcat(DenseMatrix.ones[Double](dim(X)._1, 1), X)
      w = pinv(x.t * x) * x.t * y
    }
  }


}
