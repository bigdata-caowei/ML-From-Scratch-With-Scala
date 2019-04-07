package com.scratch.ml.supervisedlearning

import breeze.linalg.DenseMatrix
import breeze.plot._
import breeze.stats.distributions.Gaussian
import com.scratch.ml.regression.LinearRegression

object LinearRegression {
  def main(args: Array[String]): Unit = {
    //构造出样本数据
    val b = DenseMatrix.rand(10, 1, Gaussian.distribution(0, 1)) * 0.01
    val x = DenseMatrix[Double, Double](1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    val y = x * 6.0 + 7.0 + b

    val regressor = new LinearRegression(gradient_descent = true)
    regressor.fit(x, y)
    println(regressor.w)
    println(regressor.trainingErrors)
    //画图
    val f = Figure()
    val p = f.subplot(0)
    p += plot(x.toDenseVector, y.toDenseVector)
    p += plot(x.toDenseVector, regressor.predict(x).toDenseVector)
    f.saveas("linear-regression.png")
  }
}
