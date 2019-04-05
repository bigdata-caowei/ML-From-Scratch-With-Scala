package com.scatch.ml.breeze

import breeze.linalg.{DenseMatrix, DenseVector}

object BreezeTest {
  def main(args: Array[String]): Unit = {
    println(DenseMatrix.zeros[Double](2,3))
    println(DenseVector.ones[Double](3))
    println(DenseVector.fill(3){3.0})
    println(DenseVector.range(0,100,1))
    println(DenseMatrix.eye[Double](3))
    println(DenseVector(1,2,3,4).t)
    println(DenseVector.tabulate(3){_*2})
    val b = DenseMatrix.rand(2,3);
    println(b)
    println(b(0,1))
  }
}
