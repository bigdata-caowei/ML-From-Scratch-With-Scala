package com.scratch.ml.classify

import breeze.linalg.{DenseMatrix, dim, sum}
import breeze.numerics.exp
import breeze.stats.distributions.Uniform
import com.scratch.ml.model.debug.Loggable

import scala.collection.mutable.ArrayBuffer

class LogisticRegression(val nIterations: Int = 4000, val learningRate: Double = 0.1, val gradientDescent: Boolean = true)
  extends Loggable {
  var w: DenseMatrix[Double] = _
  val trainingErrors = new ArrayBuffer[Double]

  def initializeWeights(nFeatures: Int): Unit = {
    val limit = 1 / Math.sqrt(nFeatures)
    logger.debug("w的上下界为 {}", limit)
    w = DenseMatrix.rand[Double](nFeatures, 1, Uniform(-limit, limit))
    logger.info("w随机初始化为{}", w)
  }

  def fit(X: DenseMatrix[Double], y: DenseMatrix[Double]): Unit = {
    if (gradientDescent) {
      val x = DenseMatrix.horzcat(DenseMatrix.ones[Double](dim(X)._1, 1), X)
      //初始化参数
      initializeWeights(dim(x)._2)
      //梯度下降迭代参数
      for (i <- 0 until nIterations) {
        val yPred = x * w
        val mse = sum((y - yPred) :* (y - yPred) * 0.5) / dim(y)._1
        trainingErrors.append(mse)
        val grad_w = x.t * (yPred - y)
        w -= learningRate * grad_w
      }

    }
  }

  def predict(X: DenseMatrix[Double]): DenseMatrix[Boolean] = {
    val x = DenseMatrix.horzcat(DenseMatrix.ones[Double](dim(X)._1, 1), X)
    return DenseMatrix.ones[Double](X.rows, 1) / (exp(-x * w) + 1.0) :> 0.5
  }
}
