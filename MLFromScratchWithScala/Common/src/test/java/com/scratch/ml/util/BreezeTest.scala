package com.scratch.ml.util

import breeze.linalg.{DenseMatrix, DenseVector, argmax, sum}
import breeze.numerics.{ceil, floor}

object BreezeTest {


  def main(args: Array[String]): Unit = {
    var x = DenseVector.zeros[Double](5)
    println(x)

    println(x(0))

    x = DenseVector(0.0, 0.2, 0.3, 0.0, 0.0)
    println(x(1))
    println(x(2 until 4))


    x(3 to 4) := 0.5
    println(x)

    x(0 to 1) := DenseVector(0.1, 0.2)
    println(x)


    val m = DenseMatrix.zeros[Int](5, 5)
    println(m)
    println((m.rows, m.cols))

    println(m(::, 1))

    m(4, ::) := DenseVector(1, 2, 3, 4, 5).t
    println(m)


    m(0 to 1, 0 to 1) := DenseMatrix((3, 1), (-1, -2))
    println(m)

    println(m :* 2)
    println(m + (m :* 2))

    println((m + 1) :> m)

    println(x dot x)

    println(sum(x))

    println(x.max)
    println(argmax(x))
    println(ceil(x))

    println(floor(x))
  }
}
