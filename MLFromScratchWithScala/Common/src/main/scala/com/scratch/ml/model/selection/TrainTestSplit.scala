package com.scratch.ml.model.selection

import breeze.linalg.{DenseMatrix, dim}
import com.scratch.ml.model.debug.Loggable

import scala.util.Random


object TrainTestSplit extends Loggable {
  def split(X: DenseMatrix[Double], y: DenseMatrix[Double], testSize: Double = 0.2, shuffle: Boolean = true):
  (DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double]) = {
    assert(X.rows == y.rows)
    val n = X.rows
    val splitIdx = (n - n * testSize).toInt

    var in = X
    var out = y

    //对数据进行shuffle，scala的shuffle操作比较麻烦,只能采取遍历循环的方式
    if (shuffle) {
      val idx = 0.until(n).toBuffer[Int]
      val shuffled = new Random().shuffle(idx)
      val xSliced = X(shuffled, ::)
      val ySliced = y(shuffled, ::)
      val (rows, cols) = dim(in)
      in = DenseMatrix.zeros(rows, cols)
      for (i <- 0 until rows) {
        for (j <- 0 until cols) {
          in(i, j) = xSliced(i, j)
        }
      }
      out = DenseMatrix.zeros(rows, 1)
      for (i <- 0 until rows) {
        out(i, 0) = ySliced(i, 0)
      }
    }

    logger.debug("总数据量为{}, 切分索引为{}", n, splitIdx)
    return (in(0 until splitIdx, ::), in(splitIdx until n, ::), out(0 until splitIdx, ::), out(splitIdx until n, ::))
  }
}
