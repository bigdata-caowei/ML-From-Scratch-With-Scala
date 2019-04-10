package com.scratch.ml.model.preprocess

import breeze.linalg.{DenseMatrix, sum, svd}

/**
  * 慎用，不要用PCA来减少variance，要使用正则化，否则可能丢失信息
  * 当参数过多，导致训练缓慢，内存消耗过大时才是使用PCA的时机
  * Andew NG
  */
object PCA {

  def transform(X: DenseMatrix[Double], k: Int): (DenseMatrix[Double], Double) = {
    val covarianceMatrix = X.t * X * (1.0 / X.rows)
    println(covarianceMatrix)
    val result = svd(covarianceMatrix)
    return (result.leftVectors(::, 0 until k), sum(result.singularValues(0 until k)) / sum(result.singularValues))
  }
}
