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


#def display_sample(display_list):
#    """Visualize side-by-side an input image, the ground-truth mask and the prediction mask
#    Args:
#        display_list (list[tf.Tensor or numpy.array]): of length 1, 2 or 3, containing the input
#        image, the ground-truth mask and the prediction mask
#    """
#    fig, axs = plt.subplots(1, len(display_list), figsize=(18, 18))
#    title = ['Input Image', 'True Mask', 'Predicted Mask']
#    for i in range(len(display_list)):
#        axs[i].set_title(title[i])
#        axs[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
#        axs[i].axis('off')
#
#def plot_predictions(model, dataset=None, sample_batch=None, num=1, save_filepaths=None):
#    """Show a sample prediction.
#    Args:
#        dataset (tf.data.Dataset, optional): dataset to take samples from
#        sample_batch (tf.Tensor, optional): a minibatch of data to plot
#        num (int): number of samples to show, default: 1
#        save_filepaths (list[path-like]): list of paths to files where to save plots for
#            every sample
#    """
#    if save_filepaths is not None:
#        assert isinstance(save_filepaths, list) and len(save_filepaths) == num
#    if dataset is not None:
#        for i, (image, mask) in enumerate(dataset.take(num)):
#            pred_raster = model.predict(image)
#            display_sample([image[0], mask, create_mask(pred_raster)])
#            if save_filepaths is not None:
#                fig = plt.gcf()
#                fig.savefig(save_filepaths[i], bbox_inches='tight', dpi=300)
#    else:
#        image, mask = sample_batch
#        pred_raster = model.predict(image)
#        pred_mask = create_mask(pred_raster)
#        for i in range(num):
#            display_sample([image[i], mask[i], pred_mask[i]])
#            if save_filepaths is not None:
#                fig = plt.gcf()
#                fig.savefig(save_filepaths[i], bbox_inches='tight', dpi=300)
