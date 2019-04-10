package com.scratch.ml.Preprocession

import breeze.linalg.DenseMatrix
import com.scratch.ml.model.preprocess.PCA

object PrincipalComponentAnalysis {

  def main(args: Array[String]): Unit = {
    val x = DenseMatrix((1.0, 2.0, 1.1), (2.0, 4.0, 6.0), (10.0, 20.0, 3.0))
    val a = new PCA().transform(x, 1)
    println(a)
  }

}
