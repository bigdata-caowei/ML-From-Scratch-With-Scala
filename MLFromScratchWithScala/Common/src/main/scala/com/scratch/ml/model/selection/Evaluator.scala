package com.scratch.ml.model.selection

import breeze.linalg.{DenseMatrix, sum}

object Evaluator {
  def accuracy(yTest: DenseMatrix[Boolean], yPredict: DenseMatrix[Boolean]): Double = {

    assert(yTest.rows == yPredict.rows)

    val result = yTest :== yPredict
    var c = result.map(x => if (x) 1 else 0)
    val b = sum(c)

    return b.toDouble / yTest.rows
    
  }
}
