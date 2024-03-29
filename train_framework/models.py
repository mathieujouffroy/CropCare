from email.mime import base
import json
import tensorflow as tf
import tensorflow.keras.layers as tfl
from collections import OrderedDict
from tensorflow import keras
from tensorflow.keras.regularizers import L2
from tensorflow.keras.initializers import glorot_uniform, random_uniform
from tensorflow.keras.applications import (
		VGG16, DenseNet201, ConvNeXtSmall, EfficientNetV2B3, Xception,
		InceptionResNetV2, InceptionV3, ResNet50V2, DenseNet201
	)
from transformers import TFConvNextModel, TFSwinModel, TFViTModel, TFCvtModel, shape_list
from train_framework.custom_inception_model import lab_two_path_inceptionresnet_v2, lab_two_path_inception_v3
from train_framework.preprocess_tensor import preprocess_image
from train_framework.interpretability import get_target_layer
from train_framework.metrics import f1_m

def unfreeze_model(model):
    # We unfreeze the model while leaving BatchNorm layers frozen
    for layer in model.layers:
        if not isinstance(layer, tfl.BatchNormalization):
            layer.trainable = True

def get_nested_base_model(model):
    last_4d = get_target_layer(model)
    if "Functional" == last_4d.__class__.__name__:
        return last_4d

def print_trainable_layers(model):
    for layer in model.layers:
        if layer.trainable:
            print (layer.name)

class LayerScale(tf.keras.layers.Layer):
    """Layer scale module.
    References:
      - https://arxiv.org/abs/2103.17239
    Args:
      init_values (float): Initial value for layer scale. Should be within
        [0, 1].
      projection_dim (int): Projection dimensionality.
    Returns:
      Tensor multiplied to the scale.
    """

    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, input_shape):
        self.gamma = tf.Variable(
            self.init_values * tf.ones((self.projection_dim,))
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_values": self.init_values,
                "projection_dim": self.projection_dim,
            }
        )
        return config

def simple_conv_model(args, mode):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE

    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process)
    """

    input_img = tf.keras.Input(shape=args.input_shape)

    prep = tfl.Lambda(preprocess_image, arguments={
                  'mean_arr': args.mean_arr, 'std_arr': args.std_arr, 'mode': mode})(input_img)

    Z1 = tfl.Conv2D(filters=8, kernel_size=(4, 4), strides=(
        1, 1), padding='same', name='conv0')(prep)
    A1 = tfl.ReLU()(Z1)
    P1 = tfl.MaxPool2D(pool_size=(8, 8), strides=(
        8, 8), padding='same', name='max_pool0')(A1)

    Z2 = tfl.Conv2D(filters=16, kernel_size=(2, 2), strides=(
        1, 1), padding='same', name='last_conv')(P1)
    A2 = tfl.ReLU()(Z2)
    P2 = tfl.MaxPool2D(pool_size=(4, 4), strides=(
        4, 4), padding='same', name='max_pool1')(A2)

    F = tfl.Flatten()(P2)

    outputs = tfl.Dense(
        units=args.n_classes, activation='softmax', name='predictions')(F)
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model



def convolutional_model_baseline(args, mode=None, l2_decay=0.0, drop_rate=0, has_batch_norm=True):
    """
    Nine-layer deep convolutional neural network. With 2 dense layer (11 total)

    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process)
    """

    input_img = tf.keras.Input(shape=args.input_shape)

    prep = tfl.Lambda(preprocess_image, arguments={'mean_arr': args.mean_arr, 'std_arr':args.std_arr, 'mode':mode})(input_img)

    Z1 = tfl.Conv2D(filters=32, kernel_size=(3, 3), padding='valid', name='conv0',
                    kernel_regularizer=L2(l2_decay))(prep)
    if (has_batch_norm):
        Z1 = keras.layers.BatchNormalization(
            axis=3, epsilon=1.001e-5)(Z1)
    A1 = tfl.ReLU()(Z1)
    P1 = tfl.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='max_pool0')(A1)

    Z2 = tfl.Conv2D(filters=16, kernel_size=(3, 3), padding='valid', name='conv1',
                    kernel_regularizer=L2(l2_decay))(P1)
    if (has_batch_norm):
        Z2 = keras.layers.BatchNormalization(
            axis=3, epsilon=1.001e-5)(Z2)
    A2 = tfl.ReLU()(Z2)
    P2 = tfl.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='max_pool1')(A2)

    Z3 = tfl.Conv2D(filters=8, kernel_size=(3, 3), padding='valid', name='last_conv',
                    kernel_regularizer=L2(l2_decay))(P2)
    if (has_batch_norm):
        Z3 = keras.layers.BatchNormalization(
            axis=3, epsilon=1.001e-5)(Z3)
    A3 = tfl.ReLU()(Z3)
    P3 = tfl.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='max_pool2')(A3)

    F = tfl.Flatten()(P3)
    Z4 = tfl.Dense(units=128, name='dense0')(F)
    if (drop_rate > 0.0):
        Z4 = tfl.Dropout(rate=drop_rate, name='dropout')(Z4)
    A4 = tfl.ReLU()(Z4)

    outputs = tfl.Dense(
        units=args.n_classes, activation='softmax', name='predictions')(A4)
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model


def alexnet_model(args, mode, l2_decay=0.0, drop_rate=0.5):
    input_img = tf.keras.Input(shape=args.input_shape)

    prep = tfl.Lambda(preprocess_image, arguments={'mean_arr': args.mean_arr, 'std_arr':args.std_arr, 'mode':mode})(input_img)
    # Layer 1
    Z1 = tfl.Conv2D(filters=96, kernel_size=(11, 11), padding='same',
                    kernel_regularizer=L2(l2_decay))(prep)
    Z1 = tfl.BatchNormalization()(Z1)
    A1 = tfl.ReLU()(Z1)
    P1 = tfl.MaxPool2D(pool_size=(2, 2))(A1)

    # Layer 2
    Z2 = tfl.Conv2D(filters=256, kernel_size=(5, 5), strides=(
        1, 1), padding='same')(P1)
    Z2 = tfl.BatchNormalization()(Z2)
    A2 = tfl.ReLU()(Z2)
    P2 = tfl.MaxPool2D(pool_size=(2, 2))(A2)

    # Layer 3
    ZP3 = tfl.ZeroPadding2D((1,1))(P2)
    Z3 = tfl.Conv2D(filters=512, kernel_size=(3, 3), padding='same')(ZP3)
    Z3 = tfl.BatchNormalization()(Z3)
    A3 = tfl.ReLU()(Z3)
    P3 = tfl.MaxPool2D(pool_size=(2, 2))(A3)

    # Layer 4
    ZP4 = tfl.ZeroPadding2D((1,1))(P3)
    Z4 = tfl.Conv2D(filters=1024, kernel_size=(3, 3), padding='same')(ZP4)
    Z4 = tfl.BatchNormalization()(Z4)
    A4 = tfl.ReLU()(Z4)

    # Layer 5
    ZP5 = tfl.ZeroPadding2D((1,1))(A4)
    Z5 = tfl.Conv2D(filters=1024, kernel_size=(3, 3), padding='same')(ZP5)
    Z5 = tfl.BatchNormalization()(Z5)
    A5 = tfl.ReLU()(Z5)
    P5 = tfl.MaxPool2D(pool_size=(2, 2))(A5)

    # Layer 6
    F = tfl.Flatten()(P5)
    Z6 = tfl.Dense(units=3072)(F)
    Z6 = tfl.BatchNormalization()(Z6)
    A6 = tfl.ReLU()(Z6)
    if (drop_rate > 0.0):
      A6 = tfl.Dropout(rate=drop_rate)(A6)

    # Layer 7
    Z7 = tfl.Dense(units=4096)(A6)
    Z7 = tfl.BatchNormalization()(Z7)
    A7 = tfl.ReLU()(Z7)
    if (drop_rate > 0.0):
      A7 = tfl.Dropout(rate=drop_rate)(A7)

    # Layer 8
    Z8 = tfl.Dense(units=4096)(A7)
    Z8 = tfl.BatchNormalization()(Z8)

    outputs = tfl.Softmax(units=args.n_classes, name='predictions')(Z8)
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model


def identity_block(X, f, filters, training=True, initializer=random_uniform):
    """
    Implementation of the identity block.

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    training -- True: Behave in training mode
                False: Behave in inference mode
    initializer -- to set up the initial weights of a layer. Equals to random uniform initializer

    Returns:
    X -- output of the identity block, tensor of shape (m, n_H, n_W, n_C)
    """

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = tfl.Conv2D(filters=F1, kernel_size=1, strides=(1, 1),
                   padding='valid', kernel_initializer=initializer(seed=0))(X)
    X = tfl.BatchNormalization(axis=3)(X, training=training)  # Default axis
    X = tfl.Activation('relu')(X)

    ## Second component of main path
    X = tfl.Conv2D(filters=F2, kernel_size=(f, f), strides=(
        1, 1), padding='same', kernel_initializer=initializer(seed=0))(X)
    X = tfl.BatchNormalization(axis=3)(X, training=training)
    X = tfl.Activation('relu')(X)

    ## Third component of main path
    X = tfl.Conv2D(filters=F3, kernel_size=(1, 1), strides=(
        1, 1), padding='valid', kernel_initializer=initializer(seed=0))(X)
    X = tfl.BatchNormalization(axis=3)(X, training=training)

    ## Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = tfl.Add()([X, X_shortcut])
    X = tfl.Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, s=2, training=True, initializer=glorot_uniform):
    """
    Implementation of the convolutional block as defined in Figure 4

    Args:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f(int): specifying the shape of the middle CONV's window for the main path
        filters(list): list of integers, defining the number of filters in the CONV layers of the main path
        s(int): specifying the stride to be used
        training(bool): If true enable training mode else enable inference mode
        initializer(): to set up the initial weights of a layer. Equals to Glorot uniform initializer,
                   also called Xavier uniform initializer.
    Returns:
        X(matrix): output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path glorot_uniform(seed=0)
    X = tfl.Conv2D(filters=F1, kernel_size=1, strides=(s, s),
                   padding='valid', kernel_initializer=initializer(seed=0))(X)
    X = tfl.BatchNormalization(axis=3)(X, training=training)
    X = tfl.Activation('relu')(X)

    ## Second component of main path
    X = tfl.Conv2D(filters=F2, kernel_size=f, strides=(1, 1),
                   padding='same', kernel_initializer=initializer(seed=0))(X)
    X = tfl.BatchNormalization(axis=3)(X, training=training)
    X = tfl.Activation('relu')(X)

    ## Third component of main path
    X = tfl.Conv2D(filters=F3, kernel_size=1, strides=(1, 1),
                   padding='valid', kernel_initializer=initializer(seed=0))(X)
    X = tfl.BatchNormalization(axis=3)(X, training=training)

    X_shortcut = tfl.Conv2D(filters=F3, kernel_size=1, strides=(
        s, s), padding='valid', kernel_initializer=initializer(seed=0))(X_shortcut)
    X_shortcut = tfl.BatchNormalization(axis=3)(X_shortcut, training=training)

    X = tfl.Add()([X, X_shortcut])
    X = tfl.Activation('relu')(X)

    return X


def Resnet50_model(args, mode):
    """
    Stage-wise implementation of the architecture of the popular ResNet50:

    Arguments:
        input_shape -- shape of the images of the dataset
        n_classes -- integer, number of classes
    Returns:
        model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = tfl.Input(args.input_shape)

    prep = tfl.Lambda(preprocess_image, arguments={'mean_arr': args.mean_arr, 'std_arr':args.std_arr, 'mode':mode})(X_input)

    # Zero-Padding
    X = tfl.ZeroPadding2D((3, 3))(prep)

    # Stage 1
    X = tfl.Conv2D(64, (7, 7), strides=(2, 2),
                   kernel_initializer=glorot_uniform(seed=0))(X)
    X = tfl.BatchNormalization(axis=3)(X)
    X = tfl.Activation('relu')(X)
    X = tfl.MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], s=1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])

    ## Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], s=2)
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])

    ## Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], s=2)
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])

    ## Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], s=2)
    X = identity_block(X, 3, [512, 512, 2048])
    X = identity_block(X, 3, [512, 512, 2048])

    X = tfl.GlobalAveragePooling2D(name='avg_pool')(X)
    X = keras.layers.Dropout(0.1)(X)

    # output layer
    X = tfl.Flatten()(X)
    X = tfl.Dense(args.n_classes, activation='softmax', name='predictions',
              kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = tf.keras.Model(inputs=X_input, outputs=X)

    return model


def set_model(args, model, mode):
    """ Set out models with their appropriate preprocessing functions. """

    if model == "simple_conv":
        model = simple_conv_model(args, mode)
    elif model ==  "conv_baseline":
        model = convolutional_model_baseline(args, mode)
    elif model == "alexnet":
        model = alexnet_model(args, mode)
    elif model == "my_Resnet50":
        model = Resnet50_model(args, mode)
    elif model == 'lab_two_path_inception_v3':
        model = lab_two_path_inception_v3(args, mode)
    elif model == 'lab_two_path_inceptionresnet_v2':
        model = lab_two_path_inceptionresnet_v2(args, mode)
    elif model == 'VGG16':
        model = VGG16
        mode = tf.keras.applications.vgg16.preprocess_input
    elif 'ResNet50V2' in model:
        model = ResNet50V2
        if mode == 'keras_imgnet':
            mode = tf.keras.applications.resnet_v2.preprocess_input
    elif 'InceptionV3' in model:
        model = InceptionV3
        if mode == 'keras_imgnet':
            mode = tf.keras.applications.inception_v3.preprocess_input
    elif model == 'Xception':
        model = Xception
        mode = tf.keras.applications.xception.preprocess_input
    elif model == 'InceptionResNetV2':
        model = InceptionResNetV2
        mode = tf.keras.applications.inception_resnet_v2.preprocess_input
    elif model == 'DenseNet201':
        model = DenseNet201
        mode = tf.keras.applications.densenet.preprocess_input
    elif model == 'EfficientNetV2B3':
        model = EfficientNetV2B3
        mode = None
    elif model == 'ConvNeXtSmall':
        model = ConvNeXtSmall
        mode = None
    return (model, mode)


def prepare_model(args, model, name, mode, t_type):
    """
    Prepare the model

    Args:
        model(keras.Model): the model to be trained
        input_shape(tuple): the shape of the input images
        n_classes(int): the number of classes
    Returns:
        model(keras.Model): the trained model
    """

    if t_type == None:
        return model

    elif t_type == "scratch":
        inputs = tfl.Input(shape=args.input_shape)
        x = preprocess_image(inputs, args.mean_arr, args.std_arr, mode)
        base_model = model(input_tensor=x, include_top=False, weights=None)
        x = base_model(x, training=True)
        print_trainable_layers(base_model)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.2)(x)
        outputs = keras.layers.Dense(
            args.n_classes, activation='softmax', name='predictions')(x)
        model = keras.Model(inputs, outputs)

    elif t_type == 'transfer':
        inputs = tfl.Input(shape=args.input_shape)
        x = preprocess_image(inputs, args.mean_arr, args.std_arr, mode)
        base_model = model(input_tensor=x, include_top=False, weights='imagenet')
        base_model.trainable = False
        # keep the BatchNormalization layers in inference mode by passing training=False
        # the batchnorm layers will not update their batch statistics.
        # This prevents the batchnorm layers from undoing all the training we've done so far.
        x = base_model(x, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.2)(x)
        outputs = keras.layers.Dense(
            args.n_classes, activation='softmax', name='predictions')(x)
        model = keras.Model(inputs, outputs)

    elif t_type == "finetune":
        args.learning_rate = 1e-5
        args.n_epochs //= 2
        base_model = get_nested_base_model(model)
        #print_trainable_layers(base_model)
        unfreeze_model(base_model)
        #print_trainable_layers(base_model)

    elif t_type == 'transformer':
        # HF -> channel first
        # heigh, width, n_chans -> n_chans, height, width
        input_shape = (args.input_shape[-1], args.input_shape[1], args.input_shape[0])
        print(f"input shape: {input_shape}")
        inputs = tfl.Input(shape=input_shape, name='pixel_values', dtype='float32')
        #p_inputs = preprocess_image(inputs, args.mean_arr, args.std_arr, mode)

        if name == 'TFViT':
            x = model.vit(inputs)[0]
            # we want to get the initial embeddig output [CLS] -> index 0 (sequence_length)
            x = x[:, 0, :]

        elif name == 'TFSwin':
            x = model.swin(inputs)[1]

        elif name == 'TFConvNexT':
            x = model.convnext(inputs)[1]

        elif name == "TFCvt":
            #x = model.cvt(inputs)[0]
            cls_token = model.cvt(inputs)[1]
            x = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm")(cls_token)
            x = tf.reduce_mean(x, axis=1)
            #batch_size, num_channels, height, width = shape_list(x)
            #x = tf.reshape(x, shape=(batch_size, num_channels, height * width))
            #x = tf.transpose(x, perm=(0, 2, 1))
            #x = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm")(x)
            #x = tf.reduce_mean(x, axis=1)
            
        outputs = keras.layers.Dense(
            args.n_classes, activation='softmax', name='predictions')(x)
        
        # hidden_state -> shape : (batch_size, sequence_length, hidden_size)
        model = keras.Model(inputs, outputs)

    return model

    
def get_models(args):
    """
    Get the models for the training and testing.
    """

    with open('resources/models_to_eval.json') as f:
        model_d = json.load(f)
    to_test = args.models
    d_subset = {key: model_d[key] for key in to_test}
    models_to_test = OrderedDict()
    for name, params in d_subset.items():
        if args.transformer:
            mode = None
            if name == 'TFViT':
                model = TFViTModel.from_pretrained("google/vit-base-patch16-224")
            elif name == 'TFSwin':
                model = TFSwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
            elif name == 'TFConvNexT':
                model = TFConvNextModel.from_pretrained("facebook/convnext-tiny-224")
            elif name == "TFCvt":
                model = TFCvtModel.from_pretrained("microsoft/cvt-13")
        else:
            if params['t_type'] == "finetune":
                mode = None
                if name.startswith('f_'):
                    name = name[2:]
                model = tf.keras.models.load_model(f"resources/best_models/cnn/{name}/model-best.h5",custom_objects={'f1_m': f1_m})
            else:
                model, mode = set_model(args, name, params['mode'])

        models_to_test[name] = prepare_model(args, model, name, mode, params['t_type'])

    return models_to_test
