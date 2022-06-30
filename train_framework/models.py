import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as tfl
from tensorflow.keras.regularizers import L2
from tensorflow.keras.initializers import random_uniform, glorot_uniform
from tensorflow.keras.applications import VGG16, ResNet50, ResNet50V2, Xception, InceptionV3, InceptionResNetV2, DenseNet201
from tensorflow.keras.applications import EfficientNetB3, EfficientNetV2B3, EfficientNetV2S, EfficientNetV2M
from custom_inception_model import *
from preprocess_tensor import preprocess_image
from transformers import TFViTModel
from transformers import ViTFeatureExtractor, TFViTForImageClassification, SwinForImageClassification


def simple_conv_model(input_shape, n_classes, mode=None):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE

    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process)
    """

    input_img = tf.keras.Input(shape=input_shape)

    #print("APPLYING CORRECT IMAGE PREPROCESSING")
    #check_preprocessing(args, valid_ds, np.uint8)
    if mode:
        input_img = preprocess_image(input_img, mode)
    #print("\n\n---- AFTER PREPROCESSING ----")
    #check_preprocessing(args, valid_ds, np.int8)

    Z1 = tfl.Conv2D(filters=8, kernel_size=(4, 4), strides=(
        1, 1), padding='same', name='conv0')(input_img)
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
        units=n_classes, activation='softmax', name='predictions')(F)
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model


def convolutional_model(input_shape, n_classes, mode=None, l2_decay=0.0, drop_rate=0, has_batch_norm=True):
    """
    Nine-layer deep convolutional neural network. With 2 dense layer (11 total)

    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process)
    """

    input_img = tf.keras.Input(shape=input_shape)
    if mode:
        input_img = preprocess_image(input_img, mode)

    Z1 = tfl.Conv2D(filters=32, kernel_size=(3, 3), padding='valid', name='conv0',
                    kernel_regularizer=L2(l2_decay))(input_img)
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
        units=n_classes, activation='softmax', name='predictions')(A4)
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model


def alexnet_model(input_shape, n_classes, mode=None, l2_decay=0.0, has_batch_norm=True, drop_rate=0):
    # resize to 227
    input_img = tf.keras.Input(shape=input_shape)
    if mode:
        input_img = preprocess_image(input_img, mode)

    # Layer 1
    Z1 = tfl.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), name='conv0', padding='valid',
                    kernel_regularizer=L2(l2_decay))(input_img)
    if has_batch_norm:
      Z1 = tfl.BatchNormalization()(Z1)
    A1 = tfl.ReLU()(Z1)
    P1 = tfl.MaxPool2D(pool_size=(3, 3), strides=(
        2, 2), name='max_pool0', padding='valid')(A1)

    # Layer 2
    Z2 = tfl.Conv2D(filters=256, kernel_size=(5, 5), strides=(
        1, 1), padding='same', name='conv1')(P1)
    if has_batch_norm:
      Z2 = tfl.BatchNormalization()(Z2)
    A2 = tfl.ReLU()(Z2)
    P2 = tfl.MaxPool2D(pool_size=(3, 3), strides=(
        2, 2), name='max_pool1', padding='valid')(A2)

    # Layer 3
    Z3 = tfl.Conv2D(filters=384, kernel_size=(3, 3), strides=(
        1, 1), padding='same', name='conv2')(P2)
    if has_batch_norm:
      Z3 = tfl.BatchNormalization()(Z3)
    A3 = tfl.ReLU()(Z3)

    # Layer 4
    Z4 = tfl.Conv2D(filters=384, kernel_size=(3, 3), strides=(
        1, 1), padding='same', name='conv3')(A3)
    if has_batch_norm:
      Z4 = tfl.BatchNormalization()(Z4)
    A4 = tfl.ReLU()(Z4)

    # Layer 5
    Z5 = tfl.Conv2D(filters=256, kernel_size=(3, 3), strides=(
        1, 1), padding='same', name='last_conv')(A4)
    if has_batch_norm:
      Z5 = tfl.BatchNormalization()(Z5)
    A5 = tfl.ReLU()(Z5)
    P5 = tfl.MaxPool2D(pool_size=(3, 3), strides=(
        2, 2), name='max_pool2', padding='valid')(A5)

    # Layer 6
    F = tfl.Flatten()(P5)
    Z6 = tfl.Dense(units=4096, name='dense0')(F)
    if has_batch_norm:
      Z6 = tfl.BatchNormalization()(Z6)
    A6 = tfl.ReLU()(Z6)
    if (drop_rate > 0.0):
      A6 = tfl.Dropout(rate=drop_rate, name='dropout0')(A6)

    # Layer 7
    Z7 = tfl.Dense(units=4096, name='dense1')(A6)
    #if has_batch_norm:
    #  Z7 = tfl.BatchNormalization()(Z7)
    A7 = tfl.ReLU()(Z7)
    if (drop_rate > 0.0):
      A7 = tfl.Dropout(rate=drop_rate, name='dropout1')(A7)

    # Layer 8
    Z8 = tfl.Dense(units=4096, name='dense2')(A7)
    #if has_batch_norm:
    #  Z8 = tfl.BatchNormalization()(Z8)
    A8 = tfl.ReLU()(Z8)
    if (drop_rate > 0.0):
      A8 = tfl.Dropout(rate=drop_rate, name='dropout2')(A8)

    outputs = tfl.Dense(
        units=n_classes, activation='softmax', name='predictions')(A8)
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model


def vgg16_model(input_shape, n_classes, mode=None, l2_decay=0.0):
    input_img = tf.keras.Input(shape=input_shape)
    if mode:
        input_img = preprocess_image(input_img, mode)

    # Block 1
    Z1 = tfl.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), name='conv0', padding='same',
                    kernel_regularizer=L2(l2_decay))(input_img)
    A1 = tfl.ReLU()(Z1)
    Z1 = tfl.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), name='conv1', padding='same',
                    kernel_regularizer=L2(l2_decay))(Z1)
    A1 = tfl.ReLU()(Z1)
    P1 = tfl.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='max_pool0')(A1)

    # Block 2
    Z2 = tfl.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), name='conv2', padding='same',
                    kernel_regularizer=L2(l2_decay))(P1)
    A2 = tfl.ReLU()(Z2)
    Z2 = tfl.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), name='conv3', padding='same',
                    kernel_regularizer=L2(l2_decay))(Z2)
    A2 = tfl.ReLU()(Z2)
    P2 = tfl.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='max_pool1')(A2)

    # Block 3
    Z3 = tfl.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), name='conv4', padding='same',
                    kernel_regularizer=L2(l2_decay))(P2)
    A3 = tfl.ReLU()(Z3)
    Z3 = tfl.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), name='conv5', padding='same',
                    kernel_regularizer=L2(l2_decay))(Z3)
    A3 = tfl.ReLU()(Z3)
    Z3 = tfl.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), name='conv6', padding='same',
                    kernel_regularizer=L2(l2_decay))(Z3)
    A3 = tfl.ReLU()(Z3)
    P3 = tfl.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='max_pool2')(A3)

    # Block 4
    Z4 = tfl.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), name='conv7', padding='same',
                    kernel_regularizer=L2(l2_decay))(P3)
    A4 = tfl.ReLU()(Z4)
    Z4 = tfl.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), name='conv8', padding='same',
                    kernel_regularizer=L2(l2_decay))(Z4)
    A4 = tfl.ReLU()(Z4)
    Z4 = tfl.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), name='conv9', padding='same',
                    kernel_regularizer=L2(l2_decay))(Z4)
    A4 = tfl.ReLU()(Z4)
    P4 = tfl.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='max_pool3')(A4)

    # Block 5
    Z5 = tfl.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), name='conv10', padding='same',
                    kernel_regularizer=L2(l2_decay))(P4)
    Z5 = tfl.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), name='conv11', padding='same',
                    kernel_regularizer=L2(l2_decay))(Z5)
    Z5 = tfl.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), name='conv12', padding='same',
                    kernel_regularizer=L2(l2_decay))(Z5)
    A5 = tfl.ReLU()(Z5)
    P5 = tfl.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='max_pool4')(A5)

    #F = tfl.Flatten()(P5)
    Z6 = tfl.Dense(4096, name='FC1')(P5)#(F)
    A6 = tfl.ReLU()(Z6)

    Z7 = tfl.Dense(4096, name='FC2')(A6)
    A7 = tfl.ReLU()(Z7)

    outputs = tfl.Dense(n_classes, activation='softmax',
                        name='predictions')(A7)

    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model


## RESNET
def identity_block(X, f, filters, training=True, initializer=random_uniform):
    """
    Implementation of the identity block as defined in Figure 4

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

    ##### SHORTCUT PATH #####
    X_shortcut = tfl.Conv2D(filters=F3, kernel_size=1, strides=(
        s, s), padding='valid', kernel_initializer=initializer(seed=0))(X_shortcut)
    X_shortcut = tfl.BatchNormalization(axis=3)(X_shortcut, training=training)

    # Final step: Add shortcut value to main path (Use this order [X, X_shortcut]), and pass it through a RELU activation
    X = tfl.Add()([X, X_shortcut])
    X = tfl.Activation('relu')(X)

    return X


def Resnet50_model(input_shape, n_classes, mode=None):
    """
    Stage-wise implementation of the architecture of the popular ResNet50:

    Arguments:
        input_shape -- shape of the images of the dataset
        n_classes -- integer, number of classes
    Returns:
        model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = tfl.Input(input_shape)
    if mode:
        X_input = preprocess_image(X_input, mode)

    # Zero-Padding
    X = tfl.ZeroPadding2D((3, 3))(X_input)

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

    X = tfl.AveragePooling2D((2, 2))(X)

    # output layer
    X = tfl.Flatten()(X)
    X = tfl.Dense(n_classes, activation='softmax', name='predictions',
                  kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = tf.keras.Model(inputs=X_input, outputs=X)

    return model


def prepare_model(model, input_shape, n_classes, mode=None, pretrained=False, finetune=False, transformer=False):
    """
    Prepare the model

    Args:
        model(keras.Model): the model to be trained
        input_shape(tuple): the shape of the input images
        n_classes(int): the number of classes
    Returns:
        model(keras.Model): the trained model
    """

    if pretrained:
        weights = 'imagenet'
    else:
        weights = None

    if transformer:
        ## TRANSFORMERS -> CHANNEL FIRST
        input_shape = (input_shape[-1], input_shape[1], input_shape[1])
        inputs = tfl.Input(shape=input_shape, name='pixel_values', dtype='float32')
        # print(model) -> transformers.models.vit.modeling_tf_vit.TFViTModel
        # get last layer output
        vit = model.vit(inputs)[0]
        # print(vit) -> KerasTensor(type_spec=TensorSpec(shape=(None, 197, 768), dtype=tf.float32, name=None), name='vit/layernorm/batchnorm/add_1:0', description="created by layer 'vit'")
        outputs = keras.layers.Dense(
            n_classes, activation='softmax', name='predictions')(vit[:, 0, :])
    else:
        base_model = model(weights=weights, input_shape=input_shape, include_top=False)

        if pretrained:
            if finetune:
                x.trainable = False
                x = x(inputs, training=False)
            else:
                x.trainable = True

        inputs = keras.Input(shape=input_shape)

        if mode:
            inputs = preprocess_image(inputs, mode)

        if pretrained:
            x = base_model(inputs, training=False)
        else:
            x = base_model(inputs, training=True)

        x = keras.layers.GlobalAveragePooling2D()(x)
        #x = keras.layers.Dropout(0.2)(x)
        outputs = keras.layers.Dense(
            n_classes, activation='softmax', name='predictions')(x)

    model = keras.Model(inputs, outputs)
    return model


def get_models(n_classes, id2label=None, imnet_weights=False):
    """
    Get the models for the training and testing.
    """
    # Test Transformers
    # Test Convnets + transformers
    input_shape = (224, 224, 3)
    label2id = {v: k for k, v in id2label.items()}
    models_dict = {
        #'baseline_sample_scale': simple_conv_model(input_shape, n_classes=n_classes, mode='sample_wise_scaling'),
        #'baseline_sample_st': simple_conv_model(input_shape, n_classes=n_classes, mode='scale_std'),
        #'basic_conv': convolutional_model(input_shape, n_classes=n_classes),
        #'basic_conv_sc_centstd': convolutional_model(input_shape, n_classes=n_classes, mode='scale_std'),
        #'basic_conv_sc_samplewise': convolutional_model(input_shape, n_classes=n_classes, mode='sample_wise_scaling'),
        #'LAB_2path_Inceptionv3_sc_samplwise': lab_two_path_inception_v3(input_shape, n_classes=n_classes, mode='sample_wise_scaling'),
        #'alexnet': alexnet_model(input_shape, n_classes=n_classes, mode='centering'),
        #'my_VGG16_sc_center': vgg16_model(input_shape, n_classes=n_classes, mode='centering'),
        #'VGG16_sc_center' : prepare_model(VGG16, input_shape, n_classes, mode='centering'),
        ## my_resnet almost identitical but very slightly less perfermant than the keras resnet
        #'my_ResNet50_sc_center': Resnet50_model(input_shape, n_classes=n_classes, mode='centering'),
        # ResNet50_sc_center' slightly less perfermant than the keras resnet
        #'ResNet50_sc_center': prepare_model(ResNet50, input_shape, n_classes, mode='centering'),
        #'ResNet50V2_sc_center': prepare_model(ResNet50V2, input_shape, n_classes, mode='centering'),
        #'InceptionV3_sc_samplwise': prepare_model(InceptionV3, input_shape, n_classes, mode='sample_wise_scaling'),
        #'DenseNet201_sc_centstd': prepare_model(DenseNet201, input_shape, n_classes, mode='scale_std'),
        #'InceptionResNetV2_sc_samplwise': prepare_model(InceptionResNetV2, input_shape, n_classes, mode='sample_wise_scaling'),
        #'EfficientNetB3': prepare_model(EfficientNetB3, input_shape, n_classes, mode='scale_std'),
        #'EfficientNetV2B3': prepare_model(EfficientNetV2B3, input_shape, n_classes, mode='scale_std'),
        #'EfficientNetV2S': prepare_model(EfficientNetV2S, input_shape, n_classes, mode='scale_std'),
        # scale_std -> as in transformers

        #'LAB_2path_InceptionResNet_V2': lab_two_path_inceptionresnet_v2(input_shape, n_classes=n_classes, mode='sample_wise_scaling'),

        'VIT': prepare_model(TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k"), input_shape, n_classes, mode='scale_std', transformer=True),

        # SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        #"VIT_HF": TFViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=n_classes, id2label=id2label,label2id=label2id)
    }
    return models_dict

# most common type of pixel scaling involves centering pixel values per-channel,
# perhaps followed by some type of normalization.
