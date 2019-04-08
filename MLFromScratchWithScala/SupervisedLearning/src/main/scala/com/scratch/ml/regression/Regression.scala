package com.scratch.ml.regression

import breeze.linalg.{DenseMatrix, dim, sum}
import breeze.stats.distributions.Uniform
import com.scratch.ml.model.debug.Loggable

import scala.collection.mutable.ArrayBuffer

class Regression(nIterations: Int, learningRate: Double) extends Loggable {
  /**
    * 回归算法的基类，作为模拟预测结果变量y 和多个独立变量之间关系的模型
    *
    * 构造方法参数：
    * n_iterations: 迭代次数
    * learning_rate: 学习率
    */

  var w: DenseMatrix[Double] = _
  val trainingErrors = new ArrayBuffer[Double]

  def initializeWeights(nFeatures: Int): Unit = {
    val limit = 1 / Math.sqrt(nFeatures)
    logger.debug("w的上下界为 {}", limit)
    w = DenseMatrix.rand[Double](nFeatures, 1, Uniform(-limit, limit))
    logger.info("w随机初始化为{}", w)
  }

  def fit(X: DenseMatrix[Double], y: DenseMatrix[Double]): Unit = {
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

  def predict(X: DenseMatrix[Double]): DenseMatrix[Double] = {
    val x = DenseMatrix.horzcat(DenseMatrix.ones[Double](dim(X)._1, 1), X)
    val y_pred = x * w
    return y_pred
  }
}
