import wandb
import random
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from train_framework.preprocess_tensor import preprocess_image


def get_imgs_table(model, test_x, test_y, nbr_imgs):
    img_dict = dict()
    for img, label in zip(test_x, test_y):
        if label not in img_dict.keys():
            img_dict[label] = []
        img_dict[label].append(img)

    imgs_table = []
    i = 0
    with open("resources/label_maps/diseases_label_map.json") as f:
        CLASS_INDEX = json.load(f)
    for unique_class, img_lst in img_dict.items():
        sample = random.sample(img_lst, nbr_imgs)
        for img in sample:
            class_name = CLASS_INDEX[str(unique_class)]
            input = tf.expand_dims(img, 0)
            probs = model.predict(input)
            pred = np.argmax(probs, axis=-1)
            pred = CLASS_INDEX[str(pred[0])]
            imgs_table.append([i, class_name, pred, img])
            i += 1

    return imgs_table

def get_target_layer(model):
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            return layer
    raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")


def make_gradcam_heatmap(args, model, m_name, img_array, pred_index=None):
    """
    GRADient-weighted Class Activation Mapping (Grad-CAM)

    We let the gradients of any target concept score (logits for any class of interest) flow
    into the final convolutional layer. We can then compute an importance score based on
    the gradients and produce a coarse localization map highlighting the important regions
    in the image for predicting that concept.
    """

    img_array = img_array[np.newaxis, :]
    model.layers[-1].activation = None
    last_4d = get_target_layer(model)
    if "Functional" == last_4d.__class__.__name__:
        last_conv = get_target_layer(last_4d)
        grad_model = tf.keras.models.Model(
            [model.inputs], [last_4d.inbound_nodes[0].output_tensors, model.output]
        )
    else:
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(
            [model.input], [model.get_layer(last_conv).output, model.output]
        )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        img_array = tf.cast(img_array, tf.float32)
        tape.watch(img_array)
        last_conv_layer_output, preds = grad_model(img_array)
        # watch the conv_output_values
        tape.watch(last_conv_layer_output)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(args, model, m_name, x_test, y_test, n_img, model_metrics_dir, alpha=0.4):
    print(model.summary())

    if args.wandb:
        TABLE_NAME = f"{m_name}_gradcam_vis"
        columns = ["id", "label", "prediction", "image", "heat_map", "img + gradcam"]
        grad_cam_table = wandb.Table(columns=columns)

    imgs_table = get_imgs_table(model, x_test, y_test, n_img)
    for elem in imgs_table:
        id = elem[0]
        label = elem[1]
        pred = elem[2]
        img = elem[-1]

        heatmap = make_gradcam_heatmap(args, model, m_name, img)
        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)
        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")
        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        # Create an image with RGB colorized heatmap
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

        if args.wandb:
            row = [id, label, pred,
                    wandb.Image(img),
                    wandb.Image(heatmap),
                    wandb.Image(superimposed_img)]
            grad_cam_table.add_data(*row)
        else:
            # Display Grad CAM
            plt.imshow(superimposed_img)
            plt.show()
            # Save the superimposed image
            superimposed_img.save(f"{model_metrics_dir}/img_{id}.jpg")


    if args.wandb:
        wandb.run.log({TABLE_NAME : grad_cam_table})
