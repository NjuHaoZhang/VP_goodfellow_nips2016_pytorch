
from six.moves import xrange
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as v1
import tensorflow.compat.v1.keras.layers as Layers

def basic_conv_lstm_cell(inputs,
                         state,
                         num_channels,
                         kernel_size=5,
                         strides=1,
                         forget_bias=1.0):
    """Basic LSTM recurrent network cell, with 2D convolution connctions.
    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.
    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.
    Args:
        inputs: input Tensor, 4D, batch x height x width x channels.
        state: state Tensor, 4D, batch x height x width x channels.
        num_channels: the number of output channels in the layer.
        kernel_size: the shape of the each convolution filter.
        forget_bias: the initial value of the forget biases.
        scope: Optional scope for variable_scope.
        reuse: whether or not the layer and the variables should be reused.
    Returns:
        a tuple of tensors representing output and the new state.
    """
    def init_state(inputs,
                state_shape,
                state_initializer=tf.zeros_initializer(),
                dtype=tf.float32):
        """Helper function to create an initial state given inputs.
        Args:
            inputs: input Tensor, at least 2D, the first dimension being batch_size
            state_shape: the shape of the state.
            state_initializer: Initializer(shape, dtype) for state Tensor.
            dtype: Optional dtype, needed when inputs is None.
        Returns:
            A tensors representing the initial state.
        """
        
            # Handle both the dynamic shape as well as the inferred shape.
        inferred_batch_size = inputs.get_shape().as_list()[0]
        dtype = inputs.dtype
        initial_state = state_initializer([inferred_batch_size] + state_shape, dtype=dtype)
        return initial_state

    spatial_size = inputs.get_shape().as_list()[1:3]
    if state is None:
        state = init_state(inputs, spatial_size + [2 * num_channels])   # (stack c, h)
    with v1.variable_scope('BasicConvLstmCell'):
        inputs.get_shape().assert_has_rank(4)
        state.get_shape().assert_has_rank(4)
        c, h = tf.split(axis=3, num_or_size_splits=2, value=state)
        inputs_h = tf.concat(axis=3, values=[inputs, h])
        # Parameters of gates are concatenated into one conv for efficiency.
        i_j_f_o = Layers.Conv2D(4 * num_channels, kernel_size=(kernel_size, kernel_size), \
            strides=(strides, strides), padding="same")(inputs_h)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(axis=3, num_or_size_splits=4, value=i_j_f_o)

        new_c = c * tf.sigmoid(f + forget_bias) + tf.sigmoid(i) * tf.tanh(j)
        new_h = tf.tanh(new_c) * tf.sigmoid(o)

    return new_h, tf.concat(axis=3, values=[new_c, new_h])

def spatial_transformer(U, theta, out_size, name='SpatialTransformer', **kwargs):
    """Spatial Transformer Layer
    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_ and edited by David Dao for Tensorflow.
    Parameters
    ----------
    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    theta: float
        The output of the
        localisation network should be [num_batch, 6].
    out_size: tuple of two ints
        The size of the output of the network (height, width)
    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py
    Notes
    -----
    To initialize the network to the identity transform init
    ``theta`` to :
        identity = np.array([[1., 0., 0.],
                             [0., 1., 0.]])
        identity = identity.flatten()
        theta = tf.Variable(initial_value=identity)
    """

    def _repeat(x, n_repeats):
        with v1.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        with v1.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            channels = tf.shape(im)[3]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0)*(width_f) / 2.0
            y = (y + 1.0)*(height_f) / 2.0

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)
            dim2 = width
            dim1 = width*height
            base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)
            base_y0 = base + y0*dim2
            base_y1 = base + y1*dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
            wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
            wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
            wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
            output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
            return output

    def _meshgrid(height, width):
        with v1.variable_scope('_meshgrid'):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
            #                         np.linspace(-1, 1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
            x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                            tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
            y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                            tf.ones(shape=tf.stack([1, width])))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            ones = tf.ones_like(x_t_flat)
            grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
            return grid

    def _transform(theta, input_dim, out_size):
        with v1.variable_scope('_transform'):
            num_batch = tf.shape(input_dim)[0]
            height = tf.shape(input_dim)[1]
            width = tf.shape(input_dim)[2]
            num_channels = tf.shape(input_dim)[3]
            theta = tf.reshape(theta, (-1, 2, 3))
            theta = tf.cast(theta, 'float32')

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            grid = _meshgrid(out_height, out_width)
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([num_batch]))
            grid = tf.reshape(grid, tf.stack([num_batch, 3, -1]))

            # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
            T_g = tf.matmul(theta, grid)
            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            input_transformed = _interpolate(
                input_dim, x_s_flat, y_s_flat,
                out_size)

            output = tf.reshape(
                input_transformed, tf.stack([num_batch, out_height, out_width, num_channels]))
            return output

    with v1.variable_scope(name):
        output = _transform(theta, U, out_size)
        return output


def batch_transformer(U, thetas, out_size, name='BatchSpatialTransformer'):
    """Batch Spatial Transformer Layer
    Parameters
    ----------
    U : float
        tensor of inputs [num_batch,height,width,num_channels]
    thetas : float
        a set of transformations for each input [num_batch,num_transforms,6]
    out_size : int
        the size of the output [out_height,out_width]
    Returns: float
        Tensor of size [num_batch*num_transforms,out_height,out_width,num_channels]
    """
    with v1.variable_scope(name):
        num_batch, num_transforms = map(int, thetas.get_shape().as_list()[:2])
        indices = [[i]*num_transforms for i in xrange(num_batch)]
        input_repeated = tf.gather(U, tf.reshape(indices, [-1]))
        return spatial_transformer(input_repeated, thetas, out_size)

    ## Utility functions
def stp_transformation(current_image, stp_input, num_masks):
    """Apply spatial transformer predictor (STP) to previous image.
    Args:
        current_image: previous image to be transformed.
        stp_input: hidden layer to be used for computing STN parameters.
        num_masks: number of masks and hence the number of STP transformations.
    Returns:
        List of images transformed by the predicted STP parameters.
    """

    identity_params = tf.convert_to_tensor(
        np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], np.float32))
    transformed = []
    for i in range(num_masks - 1):
        params = Layers.Dense(6)(stp_input) + identity_params
        transformed.append(spatial_transformer(current_image, params))

    return transformed


def cdna_transformation(current_image, cdna_input, num_masks=10, color_channels=3, 
    dna_kernel_size=5, relu_shift=1e-12):
    """Apply convolutional dynamic neural advection to previous image.
    Args:
        current_image: previous image to be transformed.
        cdna_input: hidden lyaer to be used for computing CDNA kernels.
        num_masks: the number of masks and hence the number of CDNA transformations.
        color_channels: the number of color channels in the images.
    Returns:
        List of images transformed by the predicted CDNA kernels.
    """
    batch_size = int(cdna_input.get_shape()[0])
    height = int(current_image.get_shape()[1])
    width = int(current_image.get_shape()[2])

    # Predict kernels using linear function of last hidden layer.
    cdna_kerns = Layers.Dense(dna_kernel_size * dna_kernel_size * num_masks)(cdna_input)

    # Reshape and normalize.
    cdna_kerns = tf.reshape(
        cdna_kerns, [batch_size, dna_kernel_size, dna_kernel_size, 1, num_masks])
    cdna_kerns = tf.nn.relu(cdna_kerns - relu_shift) + relu_shift
    norm_factor = tf.reduce_sum(cdna_kerns, [1, 2, 3], keepdims=True)
    cdna_kerns /= norm_factor

    # Treat the color channel dimension as the batch dimension since the same
    # transformation is applied to each color channel.
    # Treat the batch dimension as the channel dimension so that
    # depthwise_conv2d can apply a different transformation to each sample.
    cdna_kerns = tf.transpose(cdna_kerns, [1, 2, 0, 4, 3])
    cdna_kerns = tf.reshape(cdna_kerns, [dna_kernel_size, dna_kernel_size, batch_size, num_masks])
    # Swap the batch and channel dimensions.
    current_image = tf.transpose(current_image, [3, 1, 2, 0])

    # Transform image.
    transformed = tf.nn.depthwise_conv2d(current_image, cdna_kerns, [1, 1, 1, 1], 'SAME')

    # Transpose the dimensions to where they belong.
    transformed = tf.reshape(transformed, [color_channels, height, width, batch_size, num_masks])
    transformed = tf.transpose(transformed, [3, 1, 2, 0, 4])
    transformed = tf.unstack(transformed, axis=-1)
    return transformed

def dna_transformation(current_image, dna_input, dna_kernel_size=5, relu_shift=1e-12):
    """Apply dynamic neural advection to previous image.
    Args:
        current_image: maybe sampled image to be transformed.
        dna_input: hidden lyaer to be used for computing DNA transformation.
    Returns:
        List of images transformed by the predicted DNA kernels.
    """
    # Construct translated images.
    current_image_pad = tf.pad(current_image, [[0, 0], [2, 2], [2, 2], [0, 0]])
    image_height = int(current_image.get_shape()[1])
    image_width = int(current_image.get_shape()[2])

    inputs = []
    for xkern in range(dna_kernel_size):
        for ykern in range(dna_kernel_size):
            inputs.append(
                tf.expand_dims(
                    tf.slice(current_image_pad, [0, xkern, ykern, 0],
                            [-1, image_height, image_width, -1]), [3]))
    inputs = tf.concat(axis=3, values=inputs)

    # Normalize channels to 1.
    kernel = tf.nn.relu(dna_input - relu_shift) + relu_shift
    kernel = tf.expand_dims(
        kernel / tf.reduce_sum(
            kernel, [3], keepdims=True), [4])
    return tf.reduce_sum(kernel * inputs, [3], keepdims=False)
