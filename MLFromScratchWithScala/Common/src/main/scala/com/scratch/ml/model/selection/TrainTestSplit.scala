package com.scratch.ml.model.selection

import breeze.linalg.DenseMatrix
import com.scratch.ml.util.Loggable

import scala.util.Random

object a {
  def main(args: Array[String]): Unit = {

  }
}

class TrainTestSplit extends Loggable {
  def trainTestSplit(X: DenseMatrix[Double], y: DenseMatrix[Double], testSize: Double = 0.2, shuffle: Boolean = true):
  (DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double]) = {
    assert(X.rows == y.rows)
    val rows = X.rows
    if (shuffle) {
      val idx = 0.until(rows).toBuffer[Int]
      val shuffled = new Random().shuffle(idx)
    }
    val splitIdx = (rows - rows * testSize).toInt
    logger.debug("总数据量为{}, 切分索引为{}", rows, splitIdx)
    return (X(0 until splitIdx, ::), X(splitIdx until rows, ::), y(0 until splitIdx, ::), y(splitIdx until rows, ::))
  }
}
