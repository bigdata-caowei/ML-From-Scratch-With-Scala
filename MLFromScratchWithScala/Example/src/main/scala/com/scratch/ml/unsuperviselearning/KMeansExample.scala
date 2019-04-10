package com.scratch.ml.unsuperviselearning

import breeze.linalg.DenseMatrix
import com.scratch.ml.unsupervised.KMeans

object KMeansExample {
  def main(args: Array[String]): Unit = {

    val X = DenseMatrix((0.0, 0.0), (0.0, 5.0), (5.0, 0.0), (4.0, 0.0), (1.0, 4.0), (5.0, 1.0), (1.0, 1.0))


    val kMeans: KMeans = new KMeans(3)
    val clusters = kMeans.predict(X)

    clusters.foreach(println(_))
    println("*" * 100)
    println(kMeans.centroids)
  }
}
