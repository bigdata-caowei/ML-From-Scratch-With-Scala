package com.scratch.ml.util

import org.slf4j.LoggerFactory

trait Logs {
  val logger = LoggerFactory.getLogger(getClass());
}
