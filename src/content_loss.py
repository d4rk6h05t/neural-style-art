def get_content_loss(base_content, target):
  return tf.reduce_mean(tf.square(base_content - target))