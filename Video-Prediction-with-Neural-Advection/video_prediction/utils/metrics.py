import tensorflow as tf 
from video_prediction.utils.losses import l2_loss

def peak_signal_to_noise_ratio(true, pred):
	"""Image quality metric based on maximal signal power vs. power of the noise.
	Args:
		true: the ground truth image.
		pred: the predicted image.
	Returns:
		peak signal to noise ratio (PSNR)
	"""
	return 10.0 * tf.math.log(1.0 / l2_loss(true, pred)) / tf.math.log(10.0)