package com.scratch.ml.model.preprocess

import breeze.linalg.{*, DenseMatrix, DenseVector}

object Normalization {

  def normalize(mat: DenseMatrix[Double]): DenseMatrix[Double] = {
    val dividend = DenseVector.zeros[Double](mat.rows)
    for (i <- 0 until mat.rows) {
      dividend(i) = Math.sqrt(mat(i, ::) dot mat(i, ::))
    }
    return mat(::, *) :/ dividend
  }

}
