import tensorflow as tf

def poly_loss(labels, logits):
    """ Poly cross entropy loss. """
    # epsilon >=-1. =1 for first try
    # pt, CE, and Poly1 have shape [batch].
    labels = tf.cast(labels, logits.dtype)
    epsilon = 1
    pt = tf.reduce_sum(labels * tf.nn.softmax(logits), axis=-1)
    CE = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
    Poly1 = CE + epsilon * (1 - pt)
    return Poly1

def poly1_cross_entropy_label_smooth(logits, labels, epsilon):
    """ Poly cross entropy loss with alpha label smoothing """
    # epsilon >=-1.
    # one minus pt, CE, and Poly1 have shape [batch].
    alpha = 0.1
    epsilon = 1
    labels = tf.cast(labels, logits.dtype)
    num_classes = labels.get_shape().as_list()[-1]
    smooth_labels = labels * (1-alpha) + 38
    one_minus_pt = tf.reduce_sum(smooth_labels * (1 - tf.nn.softmax(logits)), axis=-1)
    CE_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=alpha, reduction=None)
    CE = CE_loss(labels, logits)
    Poly1 = CE + epsilon * one_minus_pt
    return Poly1
