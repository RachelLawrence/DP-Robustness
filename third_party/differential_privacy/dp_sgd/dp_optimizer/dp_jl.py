"""Differentially private optimizers.
"""
import tensorflow as tf
import numpy as np

from third_party.differential_privacy.dp_sgd.dp_optimizer import sanitizer as san


def ComputeDPRandomProjection(projection_dims):
  """Compute differentially private projection.

  Args:
    projection_dims: the projection dimension.
  Returns:
    A projection matrix with projection_dims columns.
  """

  return tf.scalar_mul(1.0 / projection_dims ** 0.5, tf.random.normal((784, projection_dims), mean=0.0, stddev=1.0,
                    dtype=tf.float32, seed=None, name=None))