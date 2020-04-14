import tensorflow as tf 
import tensorflow.compat.v1 as v1
from collections import OrderedDict

IMAGE_SUMMARIES = "image_summaries"
SCALAR_SUMMARIES = "scalar_summaries"

def add_image_summaries(outputs, max_outputs=1, collections=None):
    if collections is None:
        collections = [IMAGE_SUMMARIES, v1.GraphKeys.SUMMARIES]
    for name, output in outputs.items():
        if max_outputs:
            output = output[:max_outputs]
        if output.dtype != tf.uint8:        
            tf.image.convert_image_dtype(output, dtype=tf.uint8)
        if output.shape[-1] not in (1, 3):
            # these are feature maps, so just skip them
            continue
        v1.summary.image(name, output, collections=collections)

def add_scalar_summaries(losses_or_metrics, collections=None):
    if collections is None:
        collections = [SCALAR_SUMMARIES, v1.GraphKeys.SUMMARIES]
    for name, loss_or_metric in losses_or_metrics.items():
        v1.summary.scalar(name, loss_or_metric, collections=collections)

def add_summaries(outputs, collections=None):
    scalar_outputs = OrderedDict()
    image_outputs = OrderedDict()
    for name, output in outputs.items():
        if not isinstance(output, tf.Tensor):
            continue
        if output.shape.ndims == 0:
            scalar_outputs[name] = output
        elif output.shape.ndims == 4:
            image_outputs[name] = output
        elif output.shape.ndims == 5:   # sequence of images
            for idx in range(output.get_shape().as_list()[1]):
                one_name = name + "_%d"%idx
                image_outputs[one_name] = output[:, idx, :, :, :]
        else:
            continue

    add_scalar_summaries(scalar_outputs, collections=collections)
    add_image_summaries(image_outputs, collections=collections)
