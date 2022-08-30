
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as tfl
import tensorflow.keras.backend as K
from tensorflow.keras.utils import get_file
from preprocess_tensor import preprocess_image

# LAB
# smart resize is enabled.
# train shape is : (32571, 224, 224, 3)
# validation shape is : (10858, 224, 224, 3)
# test shape is : (10876, 224, 224, 3)
# Channel  0  min: 0.0         max: 1.0
# Channel  1  min: 0.19593212  max: 0.88375795
# Channel  2  min: 0.24579802  max: 0.9470912
#
# Model               1st & 2nd Layers            3rd Layer
# baseline                 32                        64
# 20 % L + 80 % AB        6 — 26                  13 — 51
# 50 % L + 50 % AB        16 — 16                 32 — 32

WEIGHTS_PATH = (
    'https://storage.googleapis.com/tensorflow/keras-applications/'
    'inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
BASE_WEIGHT_URL = ('https://github.com/fchollet/deep-learning-models/'
                   'releases/download/v0.7/')

class CopyChannels(tfl.Layer):
    """
    This layer copies channels from channel_start the number of channels given in channel_count.
    """

    def __init__(self,
                 channel_start=0,
                 channel_count=1,
                 **kwargs):
        self.channel_start = channel_start
        self.channel_count = channel_count
        super(CopyChannels, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.channel_count)

    def call(self, x):
        return x[:, :, :, self.channel_start:(self.channel_start+self.channel_count)]

    def get_config(self):
        config = {
            'channel_start': self.channel_start,
            'channel_count': self.channel_count
        }
        base_config = super(CopyChannels, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
  """Utility function to apply conv + BN.
  Args:
    x: input tensor.
    filters: filters in `Conv2D`.
    num_row: height of the convolution kernel.
    num_col: width of the convolution kernel.
    padding: padding mode in `Conv2D`.
    strides: strides in `Conv2D`.
    name: name of the ops; will become `name + '_conv'`
      for the convolution and `name + '_bn'` for the
      batch norm layer.
  Returns:
    Output tensor after applying `Conv2D` and `BatchNormalization`.
  """
  if name is not None:
    bn_name = name + '_bn'
    conv_name = name + '_conv'
  else:
    bn_name = None
    conv_name = None
  bn_axis = 3
  x = tfl.Conv2D(
      filters, (num_row, num_col),
      strides=strides,
      padding=padding,
      use_bias=False,
      name=conv_name)(
          x)
  x = tfl.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
  x = tfl.Activation('relu', name=name)(x)
  return x


def lab_two_path_inception_v3(input_shape, n_classes, include_top=True,
                              weights=None,
                              l_ratio=0.5,  # 0.2 #l_ratio in [0.2, 0.5]:
                              ab_ratio=0.5,  # 0.8  # ab_ration = 1 - l_ratio
                              model_name='two_path_inception_v3'):
    """
    Instantiates the Inception v3 architecture with 2 paths options.
    # Arguments
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_shape: mandatory input shape.
        n_classes: optional number of classes to classify images
        l_ratio: proportion dedicated to light.
        ab_ratio: proportion dedicated to color.
        model_name: model name
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    img_input = tfl.Input(shape=input_shape)

    #img_input = preprocess_image(img_input, mode)

    channel_axis = 3

    l_branch = CopyChannels(0, 1)(img_input)
    l_branch = conv2d_bn(l_branch, int(round(32*l_ratio)), 3, 3, strides=(2, 2),
                         padding='valid', name='lab_3x3l')
    l_branch = conv2d_bn(l_branch, int(round(32*l_ratio)), 3, 3,
                         padding='valid', name='lab_3x3ll')
    l_branch = conv2d_bn(l_branch, int(round(64*l_ratio)),
                         3, 3, name='lab_3x3lll')
    l_branch = tfl.MaxPooling2D(
        (3, 3), strides=(2, 2), name='lab_max_l')(l_branch)

    ab_branch = CopyChannels(1, 2)(img_input)
    ab_branch = conv2d_bn(ab_branch, int(round(32*ab_ratio)), 3, 3, strides=(2, 2),
                          padding='valid', name='lab_3x3ab')
    ab_branch = conv2d_bn(ab_branch, int(round(32*ab_ratio)), 3, 3,
                          padding='valid', name='lab_3x3abab')
    ab_branch = conv2d_bn(ab_branch, int(round(64*ab_ratio)), 3, 3,
                          name='lab_3x3ababab')
    ab_branch = tfl.MaxPooling2D(
        (3, 3), strides=(2, 2), name='lab_max_ab')(ab_branch)
    
    x = tfl.Concatenate(axis=channel_axis,
                        name='concat_first_block_lab')([l_branch, ab_branch])
    x = conv2d_bn(x, 80, 1, 1, padding='valid',  name='concat_1x1')
    x = conv2d_bn(x, 192, 3, 3, padding='valid', name='concat_3x3')
    x = tfl.MaxPooling2D((3, 3), strides=(2, 2), name='concat_max')(x)

    # mixed 0: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1, name='mixed0_1x1a')
    branch5x5 = conv2d_bn(x, 48, 1, 1, name='mixed0_1x1b')
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5, name='mixed0_5x5b')
    branch3x3dbl = conv2d_bn(x, 64, 1, 1, name='mixed0_1x1c')
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, name='mixed0_3x3c')
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, name='mixed0_3x3cc')
    branch_pool = tfl.AveragePooling2D((3, 3), strides=(1, 1), padding='same',
                                       name='mixed0_avg')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1, name='mixed0_avg1x1')
    x = tfl.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis,
        name="mixed0")

    # mixed 1: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1, name='mixed1_1x1a')
    branch5x5 = conv2d_bn(x, 48, 1, 1, name='mixed1_1x1b')
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5, name='mixed1_5x5b')
    branch3x3dbl = conv2d_bn(x, 64, 1, 1, name='mixed1_1x1c')
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, name='mixed1_3x3c')
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, name='mixed1_3x3cc')
    branch_pool = tfl.AveragePooling2D((3, 3), strides=(1, 1), padding='same',
                                       name='mixed1_avg')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1, name='mixed1_avg1x1')
    x = tfl.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1, name='mixed2_1x1a')
    branch5x5 = conv2d_bn(x, 48, 1, 1, name='mixed2_1x1b')
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5, name='mixed2_5x5b')
    branch3x3dbl = conv2d_bn(x, 64, 1, 1, name='mixed2_1x1c')
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, name='mixed2_3x3c')
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, name='mixed2_3x3bb')
    branch_pool = tfl.AveragePooling2D((3, 3), strides=(1, 1), padding='same',
                                       name='mixed2_avg')(x)
    branch_pool = conv2d_bn(branch_pool, int(64), 1, 1, name='mixed2_avg1x1')
    x = tfl.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis,
        name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(
        2, 2), padding='valid', name='mixed3_3x3a')
    branch3x3dbl = conv2d_bn(x, 64, 1, 1, name='mixed3_1x1b')
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, name='mixed3_3x3b')
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3,
                             strides=(2, 2), padding='valid')
    branch_pool = tfl.MaxPooling2D((3, 3), strides=(2, 2),
                                   name='mixed3_max')(x)
    x = tfl.concatenate(
        [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis,
        name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1, name='mixed4_1x1a')
    branch7x7 = conv2d_bn(x, 128, 1, 1, name='mixed4_1x1b')
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7, name='mixed4_1x7b')
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1, name='mixed4_7x1b')
    branch7x7dbl = conv2d_bn(x, 128, 1, 1, name='mixed4_1x1c')
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1, name='mixed4_7x1c')
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7, name='mixed4_1x7c')
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1, name='mixed4_7x1cc')
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7, name='mixed4_1x7cc')
    branch_pool = tfl.AveragePooling2D((3, 3), strides=(1, 1), padding='same',
                                       name='mixed4_avg')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1, name='mixed4_avg1x1')
    x = tfl.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis,
        name='mixed4')

    # mixed 5,  17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1, name='mixed5_1x1a')
    branch7x7 = conv2d_bn(x, 160, 1, 1, name='mixed5_1x1b')
    branch7x7 = conv2d_bn(branch7x7, 160, 1, 7, name='mixed5_1x7b')
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1, name='mixed5_7x1b')
    branch7x7dbl = conv2d_bn(x, 160, 1, 1, name='mixed5_1x1c')
    branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1, name='mixed5_7x1c')
    branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7, name='mixed5_1x7c')
    branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1, name='mixed5_7x1cc')
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7, name='mixed5_1x7cc')
    branch_pool = tfl.AveragePooling2D((3, 3), strides=(
        1, 1), padding='same', name='mixed5_avg')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1, name='mixed5_avg1x1')
    x = tfl.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis,
        name='mixed5')

    if include_top:
        # Classification block
        x = tfl.GlobalAveragePooling2D(name='avg_pool')(x)
        #x = tfl.Dropout(0.2)(x)
        x = tfl.Dense(n_classes, activation='softmax', name='predictions')(x)

    inputs = img_input
    # Create model.
    model = keras.models.Model(inputs, x, name=model_name)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model

def conv2d_bn_ir(
    x,
    filters,
    kernel_size,
    strides=1,
    padding="same",
    activation="relu",
    use_bias=False,
    name=None,
):
    """Utility function to apply conv + BN.
    Args:
      x: input tensor.
      filters: filters in `Conv2D`.
      kernel_size: kernel size as in `Conv2D`.
      strides: strides in `Conv2D`.
      padding: padding mode in `Conv2D`.
      activation: activation in `Conv2D`.
      use_bias: whether to use a bias in `Conv2D`.
      name: name of the ops; will become `name + '_ac'` for the activation
          and `name + '_bn'` for the batch norm layer.
    Returns:
      Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = tfl.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        name=name,
    )(x)
    if not use_bias:
        bn_axis =  3
        bn_name = None if name is None else name + "_bn"
        x = tfl.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(
            x
        )
    if activation is not None:
        ac_name = None if name is None else name + "_ac"
        x = tfl.Activation(activation, name=ac_name)(x)
    return x


def inception_resnet_block(x, scale, block_type, block_idx, activation="relu"):
    """Adds an Inception-ResNet block.
    This function builds 3 types of Inception-ResNet blocks mentioned
    in the paper, controlled by the `block_type` argument (which is the
    block name used in the official TF-slim implementation):
    - Inception-ResNet-A: `block_type='block35'`
    - Inception-ResNet-B: `block_type='block17'`
    - Inception-ResNet-C: `block_type='block8'`
    Args:
      x: input tensor.
      scale: scaling factor to scale the residuals (i.e., the output of passing
        `x` through an inception module) before adding them to the shortcut
        branch. Let `r` be the output from the residual branch, the output of
        this block will be `x + scale * r`.
      block_type: `'block35'`, `'block17'` or `'block8'`, determines the network
        structure in the residual branch.
      block_idx: an `int` used for generating layer names. The Inception-ResNet
        blocks are repeated many times in this network. We use `block_idx` to
        identify each of the repetitions. For example, the first
        Inception-ResNet-A block will have `block_type='block35', block_idx=0`,
        and the layer names will have a common prefix `'block35_0'`.
      activation: activation function to use at the end of the block (see
        [activations](../activations.md)). When `activation=None`, no activation
        is applied
        (i.e., "linear" activation: `a(x) = x`).
    Returns:
        Output tensor for the block.
    Raises:
      ValueError: if `block_type` is not one of `'block35'`,
        `'block17'` or `'block8'`.
    """
    if block_type == "block35":
        branch_0 = conv2d_bn_ir(x, 32, 1)
        branch_1 = conv2d_bn_ir(x, 32, 1)
        branch_1 = conv2d_bn_ir(branch_1, 32, 3)
        branch_2 = conv2d_bn_ir(x, 32, 1)
        branch_2 = conv2d_bn_ir(branch_2, 48, 3)
        branch_2 = conv2d_bn_ir(branch_2, 64, 3)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == "block17":
        branch_0 = conv2d_bn_ir(x, 192, 1)
        branch_1 = conv2d_bn_ir(x, 128, 1)
        branch_1 = conv2d_bn_ir(branch_1, 160, [1, 7])
        branch_1 = conv2d_bn_ir(branch_1, 192, [7, 1])
        branches = [branch_0, branch_1]
    elif block_type == "block8":
        branch_0 = conv2d_bn_ir(x, 192, 1)
        branch_1 = conv2d_bn_ir(x, 192, 1)
        branch_1 = conv2d_bn_ir(branch_1, 224, [1, 3])
        branch_1 = conv2d_bn_ir(branch_1, 256, [3, 1])
        branches = [branch_0, branch_1]
    else:
        raise ValueError(
            "Unknown Inception-ResNet block type. "
            'Expects "block35", "block17" or "block8", '
            "but got: " + str(block_type)
        )

    block_name = block_type + "_" + str(block_idx)
    channel_axis = 1 if K.image_data_format() == "channels_first" else 3
    mixed = tfl.Concatenate(axis=channel_axis, name=block_name + "_mixed")(
        branches
    )
    up = conv2d_bn_ir(
        mixed,
        K.int_shape(x)[channel_axis],
        1,
        activation=None,
        use_bias=True,
        name=block_name + "_conv",
    )

    x = tfl.Lambda(
        lambda inputs, scale: inputs[0] + inputs[1] * scale,
        output_shape=K.int_shape(x)[1:],
        arguments={"scale": scale},
        name=block_name,
    )([x, up])
    if activation is not None:
        x = tfl.Activation(activation, name=block_name + "_ac")(x)
    return x


def lab_two_path_inceptionresnet_v2(input_shape, n_classes, include_top=True,
                              weights=None,
                              l_ratio=0.5,  # 0.2 #l_ratio in [0.2, 0.5]:
                              ab_ratio=0.5,  # 0.8  # ab_ration = 1 - l_ratio
                              model_name='two_path_inception_v3'):
    """
    Instantiates the Inception v3 architecture with 2 paths options.
    # Arguments
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_shape: mandatory input shape.
        n_classes: optional number of classes to classify images
        l_ratio: proportion dedicated to light.
        ab_ratio: proportion dedicated to color.
        model_name: model name
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    img_input = tfl.Input(shape=input_shape)

    #img_input = preprocess_image(img_input, mode)

    channel_axis = 3

    l_branch = CopyChannels(0, 1)(img_input)
    l_branch = conv2d_bn_ir(l_branch, int(round(32*l_ratio)), 3, strides=2,
                         padding='valid', name='lab_3x3l')
    l_branch = conv2d_bn_ir(l_branch, int(round(32*l_ratio)), 3,
                         padding='valid', name='lab_3x3ll')
    l_branch = conv2d_bn_ir(l_branch, int(round(64*l_ratio)),
                         3, name='lab_3x3lll')
    l_branch = tfl.MaxPooling2D(
        (3, 3), strides=(2, 2), name='lab_max_l')(l_branch)

    ab_branch = CopyChannels(1, 2)(img_input)
    ab_branch = conv2d_bn_ir(ab_branch, int(round(32*ab_ratio)), 3, strides=2,
                          padding='valid', name='lab_3x3ab')
    ab_branch = conv2d_bn_ir(ab_branch, int(round(32*ab_ratio)), 3,
                          padding='valid', name='lab_3x3abab')
    ab_branch = conv2d_bn_ir(ab_branch, int(round(64*ab_ratio)), 3,
                          name='lab_3x3ababab')
    ab_branch = tfl.MaxPooling2D(
        (3, 3), strides=(2, 2), name='lab_max_ab')(ab_branch)
    
    x = tfl.Concatenate(axis=channel_axis,
                        name='concat_first_block_lab')([l_branch, ab_branch])
    x = conv2d_bn_ir(x, 80, 1, padding='valid',  name='concat_1x1')
    x = conv2d_bn_ir(x, 192, 3, padding='valid', name='concat_3x3')
    x = tfl.MaxPooling2D((3, 3), strides=(2, 2), name='concat_max')(x)
    
    # Mixed 5b (Inception-A block): 35 x 35 x 320
    branch_0 = conv2d_bn_ir(x, 96, 1)
    branch_1 = conv2d_bn_ir(x, 48, 1)
    branch_1 = conv2d_bn_ir(branch_1, 64, 5)
    branch_2 = conv2d_bn_ir(x, 64, 1)
    branch_2 = conv2d_bn_ir(branch_2, 96, 3)
    branch_2 = conv2d_bn_ir(branch_2, 96, 3)
    branch_pool = tfl.AveragePooling2D(3, strides=1, padding='same')(x)
    branch_pool = conv2d_bn_ir(branch_pool, 64, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = tfl.Concatenate(axis=channel_axis, name='mixed_5b')(branches)

    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(1, 11): #11
        x = inception_resnet_block(x,
                                   scale=0.17,
                                   block_type='block35',
                                   block_idx=block_idx)

    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = conv2d_bn_ir(x, 384, 3, strides=2, padding='valid')
    branch_1 = conv2d_bn_ir(x, 256, 1)
    branch_1 = conv2d_bn_ir(branch_1, 256, 3)
    branch_1 = conv2d_bn_ir(branch_1, 384, 3, strides=2, padding='valid')
    branch_pool = tfl.MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = tfl.Concatenate(axis=channel_axis, name='mixed_6a')(branches)

    # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(1, 4): # 21
        x = inception_resnet_block(x,
                                   scale=0.1,
                                   block_type='block17',
                                   block_idx=block_idx)

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    branch_0 = conv2d_bn_ir(x, 256, 1)
    branch_0 = conv2d_bn_ir(branch_0, 384, 3, strides=2, padding='valid')
    branch_1 = conv2d_bn_ir(x, 256, 1)
    branch_1 = conv2d_bn_ir(branch_1, 288, 3, strides=2, padding='valid')
    branch_2 = conv2d_bn_ir(x, 256, 1)
    branch_2 = conv2d_bn_ir(branch_2, 288, 3)
    branch_2 = conv2d_bn_ir(branch_2, 320, 3, strides=2, padding='valid')
    branch_pool = tfl.MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = tfl.Concatenate(axis=channel_axis, name='mixed_7a')(branches)

    # 9x block8 (Inception-ResNet-C block): 8 x 8 x 2080
    for block_idx in range(1, 4): # 10
        x = inception_resnet_block(x,
                                   scale=0.2,
                                   block_type='block8',
                                   block_idx=block_idx)
    x = inception_resnet_block(x,
                               scale=1.,
                               activation=None,
                               block_type='block8',
                               block_idx=10)

    # Final convolution block: 8 x 8 x 1536
    x = conv2d_bn_ir(x, 1536, 1, name='last_conv')

    if include_top:
        # Classification block
        x = tfl.GlobalAveragePooling2D(name='avg_pool')(x)
        #x = tfl.Dropout(0.2)(x)
        x = tfl.Dense(n_classes, activation='softmax', name='predictions')(x)

    inputs = img_input
    # Create model.
    model = keras.models.Model(inputs, x, name=model_name)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)


    return model