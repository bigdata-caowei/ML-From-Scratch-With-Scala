package com.scratch.ml.classify

import breeze.linalg.{DenseMatrix, dim}
import breeze.numerics.exp
import com.scratch.ml.regression.Regression

class LogisticRegression(val nIterations: Int = 4000, val learningRate: Double = 0.1, val gradientDescent: Boolean = true)
  extends Regression(nIterations, learningRate) {
  override def fit(X: DenseMatrix[Double], y: DenseMatrix[Double]): Unit = super.fit(X, y)

  override def predict(X: DenseMatrix[Double]): DenseMatrix[Double] = {
    val x = DenseMatrix.horzcat(DenseMatrix.ones[Double](dim(X)._1, 1), X)
    return DenseMatrix.ones[Double](X.rows, 1) / (exp(-x * w) + 1.0)
  }
}
