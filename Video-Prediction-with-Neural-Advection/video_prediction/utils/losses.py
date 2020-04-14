import tensorflow as tf 
import tensorflow.compat.v1 as v1
def l2_loss(pred, target):
	"""
	Can accept tensor input or tensor list input
	"""
	if isinstance(pred, list) and isinstance(target, list):
		if len(pred) != len(target):
			raise RuntimeError("pred list has len %d but target list has len %d" %(len(pred), len(target)))
		loss = 0.0
		for i in range(len(pred)):
			loss += tf.reduce_mean(tf.square(target[i] - pred[i]))/ tf.cast(tf.size(pred[i]), tf.float32)
		return loss
	else:
		return tf.reduce_mean(tf.square(target - pred))/ tf.cast(tf.size(pred), tf.float32)

