import tensorflow as tf

def poly_loss(labels, logits):
    # epsilon >=-1. =1 for first try
    # pt, CE, and Poly1 have shape [batch].
    labels = tf.cast(labels, logits.dtype)
    epsilon = 1
    pt = tf.reduce_sum(labels * tf.nn.softmax(logits), axis=-1)
    CE = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
    Poly1 = CE + epsilon * (1 - pt)
    return Poly1

def poly1_cross_entropy_label_smooth(logits, labels, epsilon):
    """ Alpha label smoothing """
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

def poly1_focal_loss(logits, labels, epsilon, gamma=2.0, alpha=0.25):
    """ Here alpha is balanced, here use of sigmoid activation """
    # epsilon >=-1.
    # p, pt, FL, weight, and Poly1 have shape [batch, num of classes].
    p = tf.math.sigmoid(logits)
    pt = labels * p + (1 - labels) * (1 - p)
    FL = focal_loss(pt, gamma, alpha)
    weight = labels * alpha + (1 - labels) * (1 - alpha)
    Poly1 = FL + epsilon * tf.math.pow(1 - pt, gamma + 1) * weight
    return Poly1

