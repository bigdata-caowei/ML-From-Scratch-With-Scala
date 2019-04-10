package com.scratch.ml.model.math

import breeze.linalg.DenseVector

object MathHelper {
  def euclideanDistance(v1: DenseVector[Double], v2: DenseVector[Double]): Double = {
    val sub = v1 - v2
    return Math.sqrt(sub dot sub)
  }
}
